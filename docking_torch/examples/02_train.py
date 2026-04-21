"""Train α / iface / charge_score on the BM4 consolidated dataset with
protein-level train / val / test split and lr selection on val.

Loads `datasets/bm4_full.h5` (built by
`scripts/build_training_dataset.py`), splits proteins
**70 % train / 15 % val / 15 % test**, runs one training pass per lr in
`--lr-grid`, picks the lr that maximises the val Hit-in-top-K metric,
and saves the winning parameters plus the held-out test set names.

The **test split is never touched** here — it is evaluated separately by
`examples/03_evaluate.py`, which reads `test_proteins` out of the ckpt.

Two knobs cover most student workflows:

    --n-proteins  how many proteins total (default 10, use "all" for 129)
    --top-k       how many top-ranked poses per protein (default 100)

Both keep the default runtime under a minute on CPU so students can
iterate. Scale up once the basics work.

Examples:
    # Quick smoke: 10 proteins, top-100 poses, 50 epochs, lr grid {0.003,0.01,0.03}
    uv run python examples/02_train.py

    # Full dataset on MPS (~5-10 min)
    uv run python examples/02_train.py --n-proteins all --top-k 2000 --device mps

    # Reproducible split (same seed -> same train/val/test sets)
    uv run python examples/02_train.py --seed 42

    # Custom lr grid
    uv run python examples/02_train.py --lr-grid 0.001,0.01,0.1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from zdock.data import list_proteins, load_training_dataset
from zdock.train import ProteinInputs, train

from _data import default_dtype, resolve_device

DEFAULT_H5 = Path(__file__).resolve().parent.parent / "datasets" / "bm4_full.h5"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def split_proteins_tvt(
    names: list[str], *, train_frac: float, val_frac: float, seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Deterministic train / val / test split on the shuffled protein list.

    `train_frac` and `val_frac` are fractions of the full list; test gets
    the remainder (1 - train_frac - val_frac). Each split is returned in
    sorted order so logs are stable across runs.
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train+val fraction must leave room for test: got "
            f"train={train_frac}, val={val_frac}"
        )
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_names = sorted(shuffled[:n_train])
    val_names = sorted(shuffled[n_train:n_train + n_val])
    test_names = sorted(shuffled[n_train + n_val:])
    return train_names, val_names, test_names


def cap_poses(p: ProteinInputs, k: int) -> ProteinInputs:
    """Return a copy of `p` keeping only the first `k` ligand poses.

    ZDOCK orders poses by raw score (highest first), so this is the
    ZDOCK top-K slice — matching the thesis evaluation convention.
    """
    k = min(k, p.lig_xyz.shape[0])
    return ProteinInputs(
        rec_xyz=p.rec_xyz, rec_radius=p.rec_radius, rec_sasa=p.rec_sasa,
        rec_atomtype_id=p.rec_atomtype_id, rec_charge_id=p.rec_charge_id,
        lig_xyz=p.lig_xyz[:k],
        lig_radius=p.lig_radius, lig_sasa=p.lig_sasa,
        lig_atomtype_id=p.lig_atomtype_id, lig_charge_id=p.lig_charge_id,
        hit_mask=p.hit_mask[:k],
        rmsd=p.rmsd[:k] if p.rmsd is not None else None,
    )


def hit_in_top_k(p: ProteinInputs, scores: torch.Tensor, k: int) -> int:
    """Number of Hit poses among the top-k scored poses (higher score better)."""
    if scores.numel() == 0:
        return 0
    k = min(k, scores.numel())
    top_idx = torch.topk(scores, k, largest=True).indices
    return int(p.hit_mask[top_idx].sum().item())


def total_hits_in_top_k(
    proteins: list[ProteinInputs],
    *,
    alpha: torch.Tensor,
    iface: torch.Tensor,
    beta: torch.Tensor,
    charge: torch.Tensor,
    top_k_eval: int,
    frame_chunk_size: int | None,
) -> int:
    """Sum of Hit-in-top-K across a protein list — the val selection metric."""
    total = 0
    with torch.no_grad():
        for p in proteins:
            s = p.call(alpha, iface, beta, charge,
                       frame_chunk_size=frame_chunk_size)
            total += hit_in_top_k(p, s, top_k_eval)
    return total


def eval_split(
    proteins: list[ProteinInputs],
    names: list[str],
    *,
    alpha: torch.Tensor,
    iface: torch.Tensor,
    beta: torch.Tensor,
    charge: torch.Tensor,
    top_k_eval: int = 10,
    frame_chunk_size: int | None = None,
) -> None:
    """Print a compact per-protein + summary line of Hit-in-top-K counts."""
    total_hit, total_with_any_hit = 0, 0
    with torch.no_grad():
        for p, name in zip(proteins, names):
            scores = p.call(
                alpha, iface, beta, charge, frame_chunk_size=frame_chunk_size,
            )
            n_hit_top = hit_in_top_k(p, scores, top_k_eval)
            total_hit += n_hit_top
            total_with_any_hit += int(p.hit_mask.any().item())
            n_hits = int(p.hit_mask.sum().item())
            print(
                f"  {name}: {n_hit_top:>3d}/{top_k_eval} in top-{top_k_eval} "
                f"({n_hits:>4d} total hits / {p.lig_xyz.shape[0]} poses)"
            )
    print(f"  -> {total_hit} hits in top-{top_k_eval} across "
          f"{total_with_any_hit}/{len(proteins)} proteins with any hit at all")


def parse_lr_grid(spec: str) -> list[float]:
    out = []
    for chunk in spec.split(","):
        s = chunk.strip()
        if not s:
            continue
        out.append(float(s))
    if not out:
        raise ValueError(f"empty lr grid: {spec!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5,
                    help=f"consolidated dataset (default: {DEFAULT_H5})")
    ap.add_argument("--n-proteins", default="10",
                    help="number of proteins to use (int or 'all'); default 10")
    ap.add_argument("--top-k", type=int, default=100,
                    help="top-K ZDOCK-ranked poses per protein (default 100)")
    ap.add_argument("--train-split", type=float, default=0.70,
                    help="train fraction of proteins (default 0.70)")
    ap.add_argument("--val-split", type=float, default=0.15,
                    help="val fraction of proteins (default 0.15). "
                         "test fraction = 1 - train - val.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr-grid", type=str, default="0.003,0.01,0.03",
                    help="comma-separated learning rates to sweep; the lr with "
                         "the highest val Hit-in-top-K is kept (default "
                         "'0.003,0.01,0.03')")
    ap.add_argument("--loss", choices=["split_mse", "rank"], default="split_mse",
                    help="training objective: 'split_mse' (Julia default, "
                         "Hit/Miss MSE) or 'rank' (ListNet on RMSD). "
                         "Default 'split_mse'.")
    ap.add_argument("--listnet-temperature", type=float, default=5.0,
                    help="temperature T (Å) for ListNet target "
                         "softmax(-rmsd/T); only used when --loss rank. "
                         "Smaller T = more peaked on the best pose; larger "
                         "T = smoother. Default 5.0.")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--top-k-eval", type=int, default=10,
                    help="rank metric: Hit count in top-K-eval (default 10)")
    ap.add_argument("--frame-chunk-size", type=int, default=64,
                    help="split the F ligand frames into chunks of this size "
                         "(gradient checkpoint) to bound peak VRAM; 0 disables "
                         "chunking (default 64, use 0 if VRAM is plentiful)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "trained_params.pt")
    args = ap.parse_args()

    if not args.h5.exists():
        raise SystemExit(
            f"dataset missing: {args.h5}\n"
            "Build it with:\n"
            "  uv run python scripts/build_training_dataset.py \\\n"
            "    --benchmark-root ../docking/decoys_bm4_zd3.0.2_6deg_fixed \\\n"
            f"    --output {args.h5}"
        )

    lr_grid = parse_lr_grid(args.lr_grid)

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    loss_desc = (
        f"rank (ListNet, T={args.listnet_temperature} Å)"
        if args.loss == "rank" else "split_mse"
    )
    print(f"device={device}  dtype={dtype}  lr_grid={lr_grid}  loss={loss_desc}")

    # 1. Pick subset of proteins.
    all_names = list_proteins(args.h5)
    if args.n_proteins == "all":
        n = len(all_names)
    else:
        n = max(1, min(int(args.n_proteins), len(all_names)))
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(all_names, n))
    print(f"{n} proteins chosen from {len(all_names)} available "
          f"(seed={args.seed}): {chosen[:6]}{'...' if n > 6 else ''}")

    # 2. 3-way split at the protein level (generalization across proteins).
    train_names, val_names, test_names = split_proteins_tvt(
        chosen, train_frac=args.train_split, val_frac=args.val_split,
        seed=args.seed,
    )
    print(f"train ({len(train_names)}): {train_names}")
    print(f"val   ({len(val_names)}): {val_names}")
    print(f"test  ({len(test_names)}): {test_names}  "
          "[held out — evaluated by 03_evaluate.py]")

    # 3. Load and cap poses. Test is NOT loaded here.
    def _load(names: list[str]) -> list[ProteinInputs]:
        if not names:
            return []
        raw = load_training_dataset(
            args.h5, device=device, dtype=dtype, protein_names=names,
            max_poses=args.top_k,
        )
        return [cap_poses(p, args.top_k) for p in raw]

    train_proteins = _load(train_names)
    val_proteins = _load(val_names)

    if not train_proteins:
        raise SystemExit("train set is empty (try a larger --n-proteins)")
    if not val_proteins:
        raise SystemExit(
            "val set is empty — can't do lr selection. Use a larger "
            "--n-proteins or a bigger --val-split."
        )

    # 4. lr grid search. Train each lr on train_proteins, score on val.
    frame_chunk_size = args.frame_chunk_size if args.frame_chunk_size > 0 else None
    beta_const = torch.tensor(3.0, device=device, dtype=dtype)

    grid_results: list[dict] = []
    best = {"val_metric": -1, "lr": None, "params": None, "history": None}
    for lr in lr_grid:
        print(f"\n--- training lr={lr} ---")
        out = train(
            train_proteins,
            n_epoch=args.epochs,
            lr=lr,
            device=device,
            dtype=dtype,
            progress_every=max(1, args.epochs // 5),
            frame_chunk_size=frame_chunk_size,
            loss=args.loss,
            listnet_temperature=args.listnet_temperature,
        )
        val_metric = total_hits_in_top_k(
            val_proteins,
            alpha=out["alpha"], iface=out["iface"], beta=beta_const,
            charge=out["charge"],
            top_k_eval=args.top_k_eval,
            frame_chunk_size=frame_chunk_size,
        )
        hist = out["history"]["loss"]
        print(f"lr={lr}  train loss: {hist[0]:.3e} -> {hist[-1]:.3e}  "
              f"val top-{args.top_k_eval} hits: {val_metric}")
        grid_results.append({
            "lr": lr,
            "val_metric": val_metric,
            "train_loss_init": hist[0],
            "train_loss_final": hist[-1],
        })
        if val_metric > best["val_metric"]:
            best = {
                "val_metric": val_metric,
                "lr": lr,
                "params": out,
                "history": hist,
            }

    print(f"\n=== grid summary (val top-{args.top_k_eval} Hit count) ===")
    for row in grid_results:
        marker = " *" if row["lr"] == best["lr"] else "  "
        print(f" {marker} lr={row['lr']:<8}  val_metric={row['val_metric']:>4d}  "
              f"final_train_loss={row['train_loss_final']:.3e}")
    print(f"selected lr = {best['lr']}  (val metric = {best['val_metric']})")

    # 5. Final report: train + val with selected params (no test eval here).
    alpha_t = best["params"]["alpha"]
    iface_t = best["params"]["iface"]
    charge_t = best["params"]["charge"]

    print(f"\n[selected params]  top-{args.top_k_eval} Hit count per protein:")
    print("train set:")
    eval_split(train_proteins, train_names,
               alpha=alpha_t, iface=iface_t, beta=beta_const, charge=charge_t,
               top_k_eval=args.top_k_eval,
               frame_chunk_size=frame_chunk_size)
    print("val set:")
    eval_split(val_proteins, val_names,
               alpha=alpha_t, iface=iface_t, beta=beta_const, charge=charge_t,
               top_k_eval=args.top_k_eval,
               frame_chunk_size=frame_chunk_size)

    # 6. Save.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "alpha": alpha_t.cpu(),
            "iface": iface_t.cpu(),
            "charge": charge_t.cpu(),
            "history": best["params"]["history"],
            "train_proteins": train_names,
            "val_proteins": val_names,
            "test_proteins": test_names,
            "selected_lr": best["lr"],
            "val_metric": best["val_metric"],
            "grid_results": grid_results,
            "top_k": args.top_k,
            "top_k_eval": args.top_k_eval,
            "seed": args.seed,
            "epochs": args.epochs,
            "loss": args.loss,
            "listnet_temperature": args.listnet_temperature,
            "h5": str(args.h5),
        },
        args.out,
    )
    print(f"\nwrote {args.out}")
    print(f"next: uv run python examples/03_evaluate.py --params {args.out}")


if __name__ == "__main__":
    main()
