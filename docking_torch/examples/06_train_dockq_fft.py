"""Train the scoring function on FFT-generated decoys with DockQ labels.

Reads an h5 produced by `scripts/build_fft_decoys.py` — every pose
comes with its own `dockq` (plus `fnat`, `i_rmsd`, `l_rmsd`). The
training split is protein-level 70/15/15; the loss can be compared
head-to-head between four modes via the `--loss` CLI flag:

    split_mse     : the Julia B2-fixed MSE loss (uses hit_mask derived
                    from dockq >= 0.23)
    rank          : ListNet on ligand RMSD (uses l_rmsd as the RMSD
                    proxy — backward-compatible with 02_train.py)
    dockq_rank    : ListNet on DockQ  (NEW)  — higher DockQ → higher
                    target prob; sign flipped vs `rank`
    dockq_margin  : hard-negative hinge on DockQ  (NEW) — enforces
                    min(positive score) ≥ max(negative score) + margin

The default `--loss dockq_rank` and `--margin-weight 0` gives the
pure ListNet-on-DockQ baseline. Users comparing losses should run
the same seed and lr grid, swapping only `--loss`.

Examples:

    # Default: DockQ rank loss on the FFT decoys produced earlier.
    uv run python examples/06_train_dockq_fft.py \
        --decoys out/fft_decoys_dockq.h5 --loss dockq_rank

    # Margin loss comparison run (same seed, same data).
    uv run python examples/06_train_dockq_fft.py \
        --decoys out/fft_decoys_dockq.h5 --loss dockq_margin

    # Cross-reference to the RMSD-based rank loss.
    uv run python examples/06_train_dockq_fft.py \
        --decoys out/fft_decoys_dockq.h5 --loss rank
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import h5py
import numpy as np
import torch

from zdock.train import ProteinInputs, train

from _data import default_dtype, resolve_device

DEFAULT_DECOYS = Path(__file__).resolve().parent.parent / "out" / "fft_decoys_dockq.h5"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def split_proteins_tvt(
    names: list[str], *,
    train_frac: float, val_frac: float, seed: int,
) -> tuple[list[str], list[str], list[str]]:
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train+val must leave room for test: got train={train_frac}, "
            f"val={val_frac}"
        )
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        sorted(shuffled[:n_train]),
        sorted(shuffled[n_train:n_train + n_val]),
        sorted(shuffled[n_train + n_val:]),
    )


def load_fft_decoy_h5(
    path: Path,
    *,
    protein_names: list[str],
    device: torch.device,
    dtype: torch.dtype,
    hit_threshold: float = 0.23,
) -> list[ProteinInputs]:
    """Load each protein's FFT-generated decoys as `ProteinInputs`.

    The FFT-decoy h5 stores `dockq`, `fnat`, `i_rmsd`, `l_rmsd` per
    pose, plus the reconstructed `lig_xyz` in the receptor's raw
    frame. This function needs to also pull the receptor and per-atom
    ligand features from the BM4 dataset that was used to build the
    decoys, because the FFT-decoy file intentionally does not
    duplicate those static fields.
    """
    # Receptor and ligand atom features are NOT in the FFT decoy h5
    # (kept compact). Pull them from bm4_full.h5 using the protein IDs.
    bm4_path = (
        Path(__file__).resolve().parent.parent / "datasets" / "bm4_full.h5"
    )
    if not bm4_path.exists():
        raise SystemExit(
            f"bm4_full.h5 not found at {bm4_path} — the FFT decoy file "
            "references atom features from BM4 and needs it available"
        )

    out: list[ProteinInputs] = []
    with h5py.File(path, "r") as f_fft, h5py.File(bm4_path, "r") as f_bm4:
        for name in protein_names:
            if name not in f_fft:
                raise KeyError(f"{path}: no group for {name}")
            if name not in f_bm4:
                raise KeyError(f"{bm4_path}: no group for {name}")
            g = f_fft[name]
            b = f_bm4[name]

            def T(g_, k, *, int_=False):
                arr = np.asarray(g_[k][()])
                t = torch.as_tensor(arr)
                if int_:
                    t = t.to(torch.int64)
                elif t.is_floating_point():
                    t = t.to(dtype)
                return t.to(device)

            lig_xyz = T(g, "lig_xyz")            # (F, N_lig, 3)
            dockq = T(g, "dockq")                # (F,)
            l_rmsd = T(g, "l_rmsd")              # (F,)
            hit_mask = (dockq >= hit_threshold)

            inputs = ProteinInputs(
                rec_xyz=T(f_bm4[name], "rec_xyz"),
                rec_radius=T(f_bm4[name], "rec_radius"),
                rec_sasa=T(f_bm4[name], "rec_sasa"),
                rec_atomtype_id=T(f_bm4[name], "rec_atomtype_id", int_=True),
                rec_charge_id=T(f_bm4[name], "rec_charge_id", int_=True),
                lig_xyz=lig_xyz,
                lig_radius=T(f_bm4[name], "lig_radius"),
                lig_sasa=T(f_bm4[name], "lig_sasa"),
                lig_atomtype_id=T(f_bm4[name], "lig_atomtype_id", int_=True),
                lig_charge_id=T(f_bm4[name], "lig_charge_id", int_=True),
                hit_mask=hit_mask,
                rmsd=l_rmsd,
                dockq=dockq,
            )
            out.append(inputs)
    return out


def mean_top_k_dockq(
    p: ProteinInputs, scores: torch.Tensor, k: int = 10,
) -> float:
    """Mean DockQ of the top-k highest-scoring poses — our main val
    metric for the DockQ-training path. Higher = better."""
    if scores.numel() == 0 or p.dockq is None:
        return 0.0
    k = min(k, scores.numel())
    top_idx = torch.topk(scores, k, largest=True).indices
    return float(p.dockq[top_idx].mean().item())


def total_top_k_dockq(
    proteins: list[ProteinInputs],
    *,
    alpha: torch.Tensor, iface: torch.Tensor,
    beta: torch.Tensor, charge: torch.Tensor,
    top_k: int, frame_chunk_size: int | None,
) -> float:
    total = 0.0
    with torch.no_grad():
        for p in proteins:
            s = p.call(alpha, iface, beta, charge,
                       frame_chunk_size=frame_chunk_size)
            total += mean_top_k_dockq(p, s, k=top_k)
    return total / max(len(proteins), 1)


def parse_lr_grid(spec: str) -> list[float]:
    out = [float(s.strip()) for s in spec.split(",") if s.strip()]
    if not out:
        raise ValueError(f"empty lr grid: {spec!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decoys", type=Path, default=DEFAULT_DECOYS,
                    help=f"FFT decoy h5 (default {DEFAULT_DECOYS})")
    ap.add_argument("--n-proteins", default="all",
                    help="number of proteins (int or 'all'); default all")
    ap.add_argument("--train-split", type=float, default=0.70)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr-grid", type=str, default="0.003,0.01,0.03")
    ap.add_argument(
        "--loss",
        choices=["split_mse", "rank", "dockq_rank", "dockq_margin"],
        default="dockq_rank",
        help="training objective (default dockq_rank). "
             "split_mse / rank match 02_train.py's modes but use "
             "the FFT-decoy h5 as input."
    )
    ap.add_argument("--listnet-temperature", type=float, default=5.0,
                    help="T for `loss=rank` target softmax(-rmsd/T). "
                         "Default 5.0 Å.")
    ap.add_argument("--dockq-temperature", type=float, default=0.2,
                    help="T for `loss=dockq_rank` target softmax(dockq/T). "
                         "Default 0.2 on [0,1] DockQ range.")
    ap.add_argument("--margin-positive-threshold", type=float, default=0.23,
                    help="DockQ >= this → positive (default 0.23, "
                         "CAPRI Acceptable). Only used with loss=dockq_margin.")
    ap.add_argument("--margin", type=float, default=1.0,
                    help="hinge margin (default 1.0). Only used with "
                         "loss=dockq_margin.")
    ap.add_argument("--hit-threshold", type=float, default=0.23,
                    help="DockQ >= this → hit_mask=True. Drives the "
                         "loss=split_mse Hit/Miss split and the val "
                         "hit-in-top-K metric (default 0.23 CAPRI).")
    ap.add_argument("--top-k-eval", type=int, default=10,
                    help="val metric: mean DockQ of top-K scored "
                         "(default 10)")
    ap.add_argument("--frame-chunk-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "trained_params_dockq_fft.pt")
    args = ap.parse_args()

    if not args.decoys.exists():
        raise SystemExit(
            f"FFT decoy h5 not found: {args.decoys}\n"
            "Build it with: uv run python scripts/build_fft_decoys.py"
        )

    lr_grid = parse_lr_grid(args.lr_grid)
    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device} dtype={dtype} loss={args.loss} "
          f"lr_grid={lr_grid}")

    with h5py.File(args.decoys, "r") as f:
        all_names = sorted(f.keys())
    if args.n_proteins == "all":
        n = len(all_names)
    else:
        n = max(1, min(int(args.n_proteins), len(all_names)))
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(all_names, n))
    print(f"{n} proteins chosen from {len(all_names)} in {args.decoys}")

    train_names, val_names, test_names = split_proteins_tvt(
        chosen,
        train_frac=args.train_split, val_frac=args.val_split,
        seed=args.seed,
    )
    print(f"train ({len(train_names)}), val ({len(val_names)}), "
          f"test ({len(test_names)})")

    train_proteins = load_fft_decoy_h5(
        args.decoys, protein_names=train_names,
        device=device, dtype=dtype, hit_threshold=args.hit_threshold,
    )
    val_proteins = load_fft_decoy_h5(
        args.decoys, protein_names=val_names,
        device=device, dtype=dtype, hit_threshold=args.hit_threshold,
    )

    frame_chunk_size = args.frame_chunk_size if args.frame_chunk_size > 0 else None
    beta_const = torch.tensor(3.0, device=device, dtype=dtype)

    grid_results = []
    best = {"val_metric": -1.0, "lr": None, "params": None}
    for lr in lr_grid:
        print(f"\n--- training lr={lr} loss={args.loss} ---")
        out = train(
            train_proteins,
            n_epoch=args.epochs, lr=lr,
            device=device, dtype=dtype,
            progress_every=max(1, args.epochs // 5),
            frame_chunk_size=frame_chunk_size,
            loss=args.loss,
            listnet_temperature=args.listnet_temperature,
            dockq_temperature=args.dockq_temperature,
            margin_positive_threshold=args.margin_positive_threshold,
            margin=args.margin,
        )
        val_metric = total_top_k_dockq(
            val_proteins,
            alpha=out["alpha"], iface=out["iface"],
            beta=beta_const, charge=out["charge"],
            top_k=args.top_k_eval,
            frame_chunk_size=frame_chunk_size,
        )
        hist = out["history"]["loss"]
        print(f"lr={lr} train loss {hist[0]:+.3e} -> {hist[-1]:+.3e}  "
              f"val mean DockQ@top-{args.top_k_eval} = {val_metric:.3f}")
        grid_results.append({
            "lr": lr, "val_metric": val_metric,
            "loss_init": hist[0], "loss_final": hist[-1],
        })
        if val_metric > best["val_metric"]:
            best = {
                "val_metric": val_metric, "lr": lr,
                "params": out,
            }

    print(f"\n=== grid summary (loss={args.loss}) ===")
    for row in grid_results:
        marker = " *" if row["lr"] == best["lr"] else "  "
        print(f" {marker} lr={row['lr']:<8} val={row['val_metric']:.3f}  "
              f"final loss={row['loss_final']:.3e}")
    print(f"selected lr = {best['lr']} val mean DockQ@{args.top_k_eval} "
          f"= {best['val_metric']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "alpha": best["params"]["alpha"].cpu(),
        "iface": best["params"]["iface"].cpu(),
        "charge": best["params"]["charge"].cpu(),
        "history": best["params"]["history"],
        "train_proteins": train_names,
        "val_proteins": val_names,
        "test_proteins": test_names,
        "selected_lr": best["lr"],
        "val_metric": best["val_metric"],
        "grid_results": grid_results,
        "seed": args.seed,
        "epochs": args.epochs,
        "loss": args.loss,
        "listnet_temperature": args.listnet_temperature,
        "dockq_temperature": args.dockq_temperature,
        "margin_positive_threshold": args.margin_positive_threshold,
        "margin": args.margin,
        "hit_threshold": args.hit_threshold,
        "top_k_eval": args.top_k_eval,
        "decoys_h5": str(args.decoys),
    }, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
