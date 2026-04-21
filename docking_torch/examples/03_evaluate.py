"""Evaluate trained BM4 parameters on the held-out test split.

Reads a ckpt saved by `examples/02_train.py`, pulls out the
`test_proteins` list (and the `h5`, `top_k` settings it was trained
with), reloads those proteins from the consolidated HDF5, and compares:

  (before) default params  (α=0.01, β=3.0, iface/charge from atomtypes)
  (after)  ckpt params     (α, iface, charge trained on train; β=3.0 fixed)

Reports per-protein:
    - Hit / Miss mean score  (before vs after)
    - Hit-in-top-K            (before vs after)
    - top rank of any Hit     (before vs after)
    - ΔScore = after - before (for Hit and Miss separately)

And a pooled before-vs-after scatter at `out/03_before_after.png`.

This script is intentionally read-only: it never updates parameters, so
running it many times on the same ckpt is cheap and deterministic.

Examples:
    # Default: evaluate the ckpt at out/trained_params.pt
    uv run python examples/03_evaluate.py

    # Custom ckpt / h5 override
    uv run python examples/03_evaluate.py --params out/my_ckpt.pt

    # Evaluate on a different split (e.g. val) recorded in the ckpt
    uv run python examples/03_evaluate.py --split val
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from zdock.atomtypes import charge_score as charge_score_lut
from zdock.atomtypes import iface_ij
from zdock.data import load_training_dataset
from zdock.train import ProteinInputs

from _data import default_dtype, resolve_device

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def default_params(device: torch.device, dtype: torch.dtype) -> dict:
    return {
        "alpha": torch.tensor(0.01, device=device, dtype=dtype),
        "beta": torch.tensor(3.0, device=device, dtype=dtype),
        "iface": iface_ij(device=device, dtype=dtype, flat=True),
        "charge": charge_score_lut(device=device, dtype=dtype),
    }


def trained_params(ckpt: dict, device: torch.device, dtype: torch.dtype) -> dict:
    def _cast(x):
        return x.to(device=device, dtype=dtype) if torch.is_tensor(x) else x
    return {
        "alpha": _cast(ckpt["alpha"]),
        "beta": _cast(ckpt["beta"]) if "beta" in ckpt
                else torch.tensor(3.0, device=device, dtype=dtype),
        "iface": _cast(ckpt["iface"]),
        "charge": _cast(ckpt["charge"]),
    }


def cap_poses(p: ProteinInputs, k: int) -> ProteinInputs:
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


def top_hit_rank(scores: torch.Tensor, hit_mask: torch.Tensor) -> int | None:
    """Best (1-based) rank of any Hit pose when sorted by score desc."""
    if not hit_mask.any():
        return None
    order = torch.argsort(scores, descending=True)
    ranks = torch.arange(1, scores.numel() + 1, device=scores.device)
    idx_of_pose = torch.empty_like(order)
    idx_of_pose[order] = ranks
    return int(idx_of_pose[hit_mask].min().item())


def hit_in_top_k(hit_mask: torch.Tensor, scores: torch.Tensor, k: int) -> int:
    if scores.numel() == 0:
        return 0
    k = min(k, scores.numel())
    top_idx = torch.topk(scores, k, largest=True).indices
    return int(hit_mask[top_idx].sum().item())


def evaluate_one(p: ProteinInputs, params: dict, top_k_eval: int,
                 frame_chunk_size: int | None) -> dict:
    with torch.no_grad():
        scores = p.call(
            params["alpha"], params["iface"], params["beta"], params["charge"],
            frame_chunk_size=frame_chunk_size,
        )
    mask = p.hit_mask
    hit_mean = float(scores[mask].mean()) if mask.any() else float("nan")
    miss_mean = float(scores[~mask].mean()) if (~mask).any() else float("nan")
    return {
        "scores": scores.detach().cpu(),
        "hit_mean": hit_mean,
        "miss_mean": miss_mean,
        "top_rank": top_hit_rank(scores, mask),
        "hit_in_top_k": hit_in_top_k(mask, scores, top_k_eval),
        "n_hits": int(mask.sum().item()),
        "n_poses": int(scores.numel()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path,
                    default=OUT_DIR / "trained_params.pt",
                    help="ckpt .pt produced by 02_train.py "
                         "(default: out/trained_params.pt)")
    ap.add_argument("--split", choices=("test", "val", "train"), default="test",
                    help="which split from the ckpt to evaluate (default: test)")
    ap.add_argument("--h5", type=Path, default=None,
                    help="override consolidated HDF5 path "
                         "(default: use the path recorded in the ckpt)")
    ap.add_argument("--top-k-eval", type=int, default=None,
                    help="top-K for Hit-in-top-K metric "
                         "(default: use the value recorded in the ckpt)")
    ap.add_argument("--frame-chunk-size", type=int, default=64,
                    help="bound peak VRAM by chunking F (0 disables)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--out-png", type=Path, default=OUT_DIR / "03_before_after.png")
    args = ap.parse_args()

    if not args.params.exists():
        raise SystemExit(
            f"{args.params} not found — run examples/02_train.py first."
        )

    ckpt = torch.load(args.params, map_location="cpu", weights_only=False)

    split_key = f"{args.split}_proteins"
    if split_key not in ckpt:
        raise SystemExit(
            f"ckpt does not record '{split_key}'. "
            "Was it produced by the current 02_train.py? "
            f"Keys present: {sorted(ckpt.keys())}"
        )
    split_names = list(ckpt[split_key])
    if not split_names:
        raise SystemExit(f"{split_key} is empty in the ckpt — nothing to evaluate.")

    h5_path = args.h5 if args.h5 is not None else Path(ckpt.get("h5", ""))
    if not h5_path or not h5_path.exists():
        raise SystemExit(
            f"HDF5 not found at {h5_path}. Pass --h5 explicitly."
        )

    top_k = int(ckpt.get("top_k", 100))
    top_k_eval = args.top_k_eval if args.top_k_eval is not None \
        else int(ckpt.get("top_k_eval", 10))
    frame_chunk_size = args.frame_chunk_size if args.frame_chunk_size > 0 else None

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device}  dtype={dtype}  split={args.split}  "
          f"top_k={top_k}  top_k_eval={top_k_eval}")
    print(f"ckpt={args.params}  selected_lr={ckpt.get('selected_lr')}  "
          f"val_metric={ckpt.get('val_metric')}")
    print(f"{args.split} proteins ({len(split_names)}): {split_names}")

    raw = load_training_dataset(
        h5_path, device=device, dtype=dtype, protein_names=split_names,
        max_poses=top_k,
    )
    proteins = [cap_poses(p, top_k) for p in raw]

    before = default_params(device, dtype)
    after = trained_params(ckpt, device, dtype)

    all_before, all_after, all_hit = [], [], []
    total_hit_before, total_hit_after, n_with_any_hit = 0, 0, 0
    print(
        f"\n{'protein':<8} {'phase':<6} {'hit μ':>10} {'miss μ':>10} "
        f"{'Δhit':>9} {'Δmiss':>9} {'top':>5} {'top-' + str(top_k_eval):>8}"
    )
    for name, p in zip(split_names, proteins):
        b = evaluate_one(p, before, top_k_eval, frame_chunk_size)
        a = evaluate_one(p, after, top_k_eval, frame_chunk_size)
        dh = a["hit_mean"] - b["hit_mean"]
        dm = a["miss_mean"] - b["miss_mean"]
        br = b["top_rank"] if b["top_rank"] is not None else "-"
        ar = a["top_rank"] if a["top_rank"] is not None else "-"
        print(
            f"{name:<8} {'before':<6} {b['hit_mean']:>10.3f} "
            f"{b['miss_mean']:>10.3f} {'':>9} {'':>9} {br!s:>5} "
            f"{b['hit_in_top_k']:>8d}"
        )
        print(
            f"{'':<8} {'after':<6} {a['hit_mean']:>10.3f} "
            f"{a['miss_mean']:>10.3f} {dh:>9.3f} {dm:>9.3f} {ar!s:>5} "
            f"{a['hit_in_top_k']:>8d}"
        )
        all_before.append(b["scores"].numpy())
        all_after.append(a["scores"].numpy())
        all_hit.append(p.hit_mask.detach().cpu().numpy())
        total_hit_before += b["hit_in_top_k"]
        total_hit_after += a["hit_in_top_k"]
        n_with_any_hit += int(p.hit_mask.any().item())

    print(
        f"\nsummary ({args.split}): "
        f"top-{top_k_eval} hits  before={total_hit_before}  "
        f"after={total_hit_after}  "
        f"Δ={total_hit_after - total_hit_before:+d}  "
        f"(across {n_with_any_hit}/{len(proteins)} proteins with any hit)"
    )
    print("top = best (1-based) rank of any Hit pose")

    # Scatter of pooled per-pose scores.
    before_flat = np.concatenate(all_before)
    after_flat = np.concatenate(all_after)
    hit_flat = np.concatenate(all_hit).astype(bool)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(before_flat[~hit_flat], after_flat[~hit_flat],
               s=12, c="lightgrey", label="Miss", edgecolor="none", alpha=0.7)
    ax.scatter(before_flat[hit_flat], after_flat[hit_flat],
               s=28, c="tab:blue", label="Hit", edgecolor="k", linewidth=0.5)
    lo = float(min(before_flat.min(), after_flat.min()))
    hi = float(max(before_flat.max(), after_flat.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("score (before: default params)")
    ax.set_ylabel("score (after: trained params)")
    ax.set_title(f"{args.split} split: before vs after  "
                 f"({len(proteins)} proteins, {before_flat.size} poses)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
