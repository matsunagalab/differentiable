"""Student-friendly training + evaluation on the BM4 consolidated dataset.

Loads `datasets/bm4_full.h5` (built by
`scripts/build_training_dataset.py`), splits proteins 70 % train / 30 %
test, trains α / β / iface / charge_score on the train set, and reports
a simple ranking metric (Hit count in top-K) on both sets.

Two knobs cover most student workflows:

    --n-proteins  how many proteins total (default 10, use "all" for 129)
    --top-k       how many top-ranked poses per protein (default 100)

Both keep the default runtime under a minute on CPU so students can
iterate. Scale up once the basics work.

Examples:
    # Quick smoke: 10 proteins, top-100 poses, 50 epochs
    uv run python examples/04_train_on_bm4.py

    # Full dataset on MPS (~5-10 min)
    uv run python examples/04_train_on_bm4.py --n-proteins all --top-k 2000 --device mps

    # Reproducible split (same seed -> same train/test sets)
    uv run python examples/04_train_on_bm4.py --seed 42
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


def split_proteins(
    names: list[str], *, train_frac: float, seed: int,
) -> tuple[list[str], list[str]]:
    """Deterministic train / test split on the sorted protein list."""
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    train_names = sorted(shuffled[:n_train])
    test_names = sorted(shuffled[n_train:])
    return train_names, test_names


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5,
                    help=f"consolidated dataset (default: {DEFAULT_H5})")
    ap.add_argument("--n-proteins", default="10",
                    help="number of proteins to use (int or 'all'); default 10")
    ap.add_argument("--top-k", type=int, default=100,
                    help="top-K ZDOCK-ranked poses per protein (default 100)")
    ap.add_argument("--split", type=float, default=0.7,
                    help="train fraction (default 0.7)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--top-k-eval", type=int, default=10,
                    help="rank metric: Hit count in top-K-eval (default 10)")
    ap.add_argument("--frame-chunk-size", type=int, default=64,
                    help="split the F ligand frames into chunks of this size "
                         "(gradient checkpoint) to bound peak VRAM; 0 disables "
                         "chunking (default 64, use 0 if VRAM is plentiful)")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "trained_params_bm4.pt")
    args = ap.parse_args()

    if not args.h5.exists():
        raise SystemExit(
            f"dataset missing: {args.h5}\n"
            "Build it with:\n"
            "  uv run python scripts/build_training_dataset.py \\\n"
            "    --benchmark-root ../docking/decoys_bm4_zd3.0.2_6deg_fixed \\\n"
            f"    --output {args.h5}"
        )

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device}  dtype={dtype}")

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

    # 2. Split 70/30 at the protein level (generalization across proteins).
    train_names, test_names = split_proteins(
        chosen, train_frac=args.split, seed=args.seed,
    )
    print(f"train ({len(train_names)}): {train_names}")
    print(f"test  ({len(test_names)}): {test_names}")

    # 3. Load and cap poses. max_poses=args.top_k ensures the full 54k-pose
    # trajectory never touches device memory; cap_poses below is idempotent
    # when the h5 read already gave us <= top_k poses.
    def _load(names: list[str]) -> list[ProteinInputs]:
        if not names:
            return []
        raw = load_training_dataset(
            args.h5, device=device, dtype=dtype, protein_names=names,
            max_poses=args.top_k,
        )
        return [cap_poses(p, args.top_k) for p in raw]

    train_proteins = _load(train_names)
    test_proteins = _load(test_names)

    if not train_proteins:
        raise SystemExit("train set is empty (try a larger --n-proteins)")

    # 4. Train.
    frame_chunk_size = args.frame_chunk_size if args.frame_chunk_size > 0 else None
    out = train(
        train_proteins,
        n_epoch=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
        progress_every=max(1, args.epochs // 10),
        frame_chunk_size=frame_chunk_size,
    )
    hist = out["history"]["loss"]
    print(f"\ntrain loss: {hist[0]:.3e} -> {hist[-1]:.3e} "
          f"(reduction {(1 - hist[-1] / hist[0]) * 100:.1f}%)")

    # 5. Evaluate before/after on both splits.
    alpha_t = out["alpha"]
    iface_t = out["iface"]
    beta_t = out["beta"]
    charge_t = out["charge"]
    print(f"\n[trained params] top-{args.top_k_eval} Hit count per protein:")
    print("train set:")
    eval_split(train_proteins, train_names,
               alpha=alpha_t, iface=iface_t, beta=beta_t, charge=charge_t,
               top_k_eval=args.top_k_eval,
               frame_chunk_size=frame_chunk_size)
    if test_proteins:
        print("test set (generalization):")
        eval_split(test_proteins, test_names,
                   alpha=alpha_t, iface=iface_t, beta=beta_t, charge=charge_t,
                   top_k_eval=args.top_k_eval,
                   frame_chunk_size=frame_chunk_size)

    # 6. Save.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "alpha": alpha_t.cpu(),
            "beta": beta_t.cpu(),
            "iface": iface_t.cpu(),
            "charge": charge_t.cpu(),
            "history": out["history"],
            "train_proteins": train_names,
            "test_proteins": test_names,
            "top_k": args.top_k,
            "seed": args.seed,
            "epochs": args.epochs,
        },
        args.out,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
