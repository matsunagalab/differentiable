"""Evaluate a trained scorer on an FFT-decoy h5 with DockQ labels.

Reads the checkpoint written by `examples/06_train_dockq_fft.py` and
the FFT-decoy h5 it was trained on. Computes before-vs-after scoring
statistics on the held-out test split (or `--split val`/`train`):

    * top-1 DockQ              — quality of the single highest-scored pose
    * best DockQ in top-K      — best quality among the K highest-scored
    * CAPRI tier of top-1      — which of incorrect / acceptable / medium / high
    * success rates            — % of test proteins achieving ≥ acceptable /
                                   medium / high in their top-1

Contrast with `examples/03_evaluate.py`:
    03_evaluate.py works on BM4 h5 + Hit-in-top-K metric, which is the
    canonical thesis evaluator. This script (07) is its DockQ-aware
    cousin for the FFT + DockQ training workflow.

Examples:

    # Evaluate the default ckpt on its held-out test set
    uv run python examples/07_evaluate_dockq.py

    # Compare dockq_rank vs dockq_margin on the same test set
    uv run python examples/07_evaluate_dockq.py \
        --params out/trained_params_dockq_fft.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from zdock.dockq import CAPRI_ACCEPTABLE, CAPRI_MEDIUM, CAPRI_HIGH, capri_tier
from zdock.train import ProteinInputs

# Reuse the loader from 06 — they read the same h5 schema.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _data import default_dtype, resolve_device  # noqa: E402

_TIER_NAMES = ["incorrect", "acceptable", "medium", "high"]


def load_fft_decoy_h5(
    path: Path,
    *,
    protein_names: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> list[ProteinInputs]:
    bm4_path = (
        Path(__file__).resolve().parent.parent / "datasets" / "bm4_full.h5"
    )
    out: list[ProteinInputs] = []
    with h5py.File(path, "r") as f_fft, h5py.File(bm4_path, "r") as f_bm4:
        for name in protein_names:
            if name not in f_fft:
                raise KeyError(f"{path}: no group for {name}")
            g, b = f_fft[name], f_bm4[name]

            def T(g_, k, *, int_=False):
                t = torch.as_tensor(np.asarray(g_[k][()]))
                if int_:
                    t = t.to(torch.int64)
                elif t.is_floating_point():
                    t = t.to(dtype)
                return t.to(device)

            dockq = T(g, "dockq")
            out.append(ProteinInputs(
                rec_xyz=T(b, "rec_xyz"),
                rec_radius=T(b, "rec_radius"),
                rec_sasa=T(b, "rec_sasa"),
                rec_atomtype_id=T(b, "rec_atomtype_id", int_=True),
                rec_charge_id=T(b, "rec_charge_id", int_=True),
                lig_xyz=T(g, "lig_xyz"),
                lig_radius=T(b, "lig_radius"),
                lig_sasa=T(b, "lig_sasa"),
                lig_atomtype_id=T(b, "lig_atomtype_id", int_=True),
                lig_charge_id=T(b, "lig_charge_id", int_=True),
                hit_mask=(dockq >= CAPRI_ACCEPTABLE),
                rmsd=T(g, "l_rmsd"),
                dockq=dockq,
            ))
    return out


def evaluate_one(
    name: str, p: ProteinInputs,
    *,
    alpha_before: torch.Tensor, iface_before: torch.Tensor,
    charge_before: torch.Tensor,
    alpha_after: torch.Tensor, iface_after: torch.Tensor,
    charge_after: torch.Tensor,
    beta: torch.Tensor,
    top_k: int, frame_chunk_size: int | None,
) -> dict:
    with torch.no_grad():
        s_before = p.call(
            alpha_before, iface_before, beta, charge_before,
            frame_chunk_size=frame_chunk_size,
        )
        s_after = p.call(
            alpha_after, iface_after, beta, charge_after,
            frame_chunk_size=frame_chunk_size,
        )

    def top_k_dockq(scores):
        k = min(top_k, scores.numel())
        idx_top = torch.topk(scores, k, largest=True).indices
        return p.dockq[idx_top]

    dq_top_before = top_k_dockq(s_before)
    dq_top_after = top_k_dockq(s_after)
    return {
        "name": name,
        "top1_dockq_before": float(dq_top_before[0].item()),
        "top1_dockq_after": float(dq_top_after[0].item()),
        "best_dockq_in_topk_before": float(dq_top_before.max().item()),
        "best_dockq_in_topk_after": float(dq_top_after.max().item()),
        "top1_tier_before": int(capri_tier(dq_top_before[:1])[0].item()),
        "top1_tier_after": int(capri_tier(dq_top_after[:1])[0].item()),
        "best_available_dockq": float(p.dockq.max().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=Path,
                    default=Path(__file__).resolve().parent.parent
                          / "out" / "trained_params_dockq_fft.pt",
                    help="checkpoint from 06_train_dockq_fft.py")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test"],
                    help="which split of the ckpt to evaluate "
                         "(default: test)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="K for best-DockQ-in-top-K (default 10)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--frame-chunk-size", type=int, default=64)
    args = ap.parse_args()

    if not args.params.exists():
        raise SystemExit(
            f"checkpoint not found: {args.params}\n"
            "Train with: uv run python examples/06_train_dockq_fft.py"
        )

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    ckpt = torch.load(args.params, weights_only=False, map_location="cpu")

    proteins_for_split = ckpt[f"{args.split}_proteins"]
    decoys_h5 = Path(ckpt["decoys_h5"])
    if not decoys_h5.exists():
        raise SystemExit(
            f"decoy h5 referenced by ckpt not found: {decoys_h5}"
        )

    print(f"device={device} dtype={dtype}  split={args.split}  "
          f"loss={ckpt.get('loss')}  ckpt={args.params}")
    print(f"{len(proteins_for_split)} proteins from {decoys_h5}")

    alpha_after = ckpt["alpha"].to(device=device, dtype=dtype)
    iface_after = ckpt["iface"].to(device=device, dtype=dtype)
    charge_after = ckpt["charge"].to(device=device, dtype=dtype)

    # Before = Julia LUT defaults (same convention as 03_evaluate.py).
    from zdock.atomtypes import iface_ij as default_iface_ij
    from zdock.atomtypes import charge_score as default_charge_score
    alpha_before = torch.tensor(0.01, device=device, dtype=dtype)
    iface_before = default_iface_ij(device=device, dtype=dtype, flat=True)
    charge_before = default_charge_score(device=device, dtype=dtype)
    beta = torch.tensor(3.0, device=device, dtype=dtype)

    proteins = load_fft_decoy_h5(
        decoys_h5, protein_names=proteins_for_split,
        device=device, dtype=dtype,
    )

    chunk = args.frame_chunk_size if args.frame_chunk_size > 0 else None
    rows = []
    for name, p in zip(proteins_for_split, proteins):
        rows.append(evaluate_one(
            name, p,
            alpha_before=alpha_before, iface_before=iface_before,
            charge_before=charge_before,
            alpha_after=alpha_after, iface_after=iface_after,
            charge_after=charge_after,
            beta=beta, top_k=args.top_k, frame_chunk_size=chunk,
        ))

    # Table
    print()
    print(f"  {'protein':>8}  {'top-1 DockQ':>12}      "
          f"{'best DockQ in top-%d' % args.top_k:>22}      "
          f"{'tier (top-1)':>14}  {'best avail':>10}")
    print(f"  {'':>8}  {'before → after':>12}      "
          f"{'before → after':>22}      "
          f"{'before → after':>14}  {'DockQ':>10}")
    print("  " + "-" * 88)
    for r in rows:
        tb, ta = _TIER_NAMES[r["top1_tier_before"]], _TIER_NAMES[r["top1_tier_after"]]
        print(
            f"  {r['name']:>8}  "
            f"{r['top1_dockq_before']:>5.2f} → {r['top1_dockq_after']:<5.2f}  "
            f"      {r['best_dockq_in_topk_before']:>5.2f} → "
            f"{r['best_dockq_in_topk_after']:<5.2f}"
            f"            {tb[:6]:>6} → {ta[:6]:<6}"
            f"     {r['best_available_dockq']:.2f}"
        )

    # Summary
    def rate(field_name: str, threshold: float) -> tuple[float, float]:
        before_hits = sum(1 for r in rows if r[f"{field_name}_before"] >= threshold)
        after_hits = sum(1 for r in rows if r[f"{field_name}_after"] >= threshold)
        n = max(len(rows), 1)
        return before_hits / n, after_hits / n

    def mean(field_name: str) -> tuple[float, float]:
        before = np.mean([r[f"{field_name}_before"] for r in rows]) if rows else 0.0
        after = np.mean([r[f"{field_name}_after"] for r in rows]) if rows else 0.0
        return float(before), float(after)

    print("\nsummary (N={}):".format(len(rows)))
    mb, ma = mean("top1_dockq")
    print(f"  mean top-1 DockQ:                  before={mb:.3f}  after={ma:.3f}  Δ={ma-mb:+.3f}")
    mb, ma = mean("best_dockq_in_topk")
    print(f"  mean best DockQ in top-{args.top_k}:           before={mb:.3f}  after={ma:.3f}  Δ={ma-mb:+.3f}")
    for tname, thresh in [
        ("acceptable", CAPRI_ACCEPTABLE),
        ("medium",     CAPRI_MEDIUM),
        ("high",       CAPRI_HIGH),
    ]:
        rb, ra = rate("top1_dockq", thresh)
        print(f"  top-1 {tname:>10} rate (DockQ ≥ {thresh:.2f}): "
              f"before={rb:.2%}  after={ra:.2%}  Δ={ra-rb:+.2%}")


if __name__ == "__main__":
    main()
