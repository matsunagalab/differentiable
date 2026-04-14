"""Before / after evaluation of trained parameters.

Scores each protein's poses with (a) the untrained defaults and (b) the
parameters from `out/trained_params.pt` (or any `.pt` produced by
02_train.py), then reports:

    - Hit / Miss mean score (before vs after) per protein
    - top rank of any Hit pose (before vs after)
    - ΔScore = after - before, separately for Hit / Miss
    - a before-vs-after scatter at out/03_before_after.png

Dataset extensibility is the same as 02_train.py: pass `--proteins …`
pointing at IDs defined in `examples/_data.py::DATASETS`. Test-set
evaluation (F-3) requires only that the test IDs be added to `DATASETS`.

Run:
    uv run python examples/03_evaluate.py --params out/trained_params.pt
    uv run python examples/03_evaluate.py --params out/trained_params.pt --proteins 1KXQ
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

from _data import default_dtype, load_protein, resolve_device

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def default_params(device: torch.device, dtype: torch.dtype) -> dict:
    return {
        "alpha": torch.tensor(0.01, device=device, dtype=dtype),
        "beta": torch.tensor(3.0, device=device, dtype=dtype),
        "iface": iface_ij(device=device, dtype=dtype, flat=True),
        "charge": charge_score_lut(device=device, dtype=dtype),
    }


def trained_params(path: Path, device: torch.device, dtype: torch.dtype) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    def _cast(x):
        return x.to(device=device, dtype=dtype) if torch.is_tensor(x) else x
    return {
        "alpha": _cast(ckpt["alpha"]),
        "beta": _cast(ckpt["beta"]),
        "iface": _cast(ckpt["iface"]),
        "charge": _cast(ckpt["charge"]),
    }


def top_hit_rank(scores: torch.Tensor, hit_mask: torch.Tensor) -> int | None:
    """Highest (1-based) rank at which any Hit pose appears when sorted by
    score descending. None if there are no Hits."""
    if not hit_mask.any():
        return None
    order = torch.argsort(scores, descending=True)
    ranks = torch.arange(1, scores.numel() + 1, device=scores.device)
    idx_of_pose = torch.empty_like(order)
    idx_of_pose[order] = ranks
    return int(idx_of_pose[hit_mask].min().item())


def evaluate_one(lp, params: dict) -> dict:
    with torch.no_grad():
        scores = lp.inputs.call(
            params["alpha"], params["iface"], params["beta"], params["charge"]
        )
    mask = lp.inputs.hit_mask
    hit_mean = float(scores[mask].mean()) if mask.any() else float("nan")
    miss_mean = float(scores[~mask].mean()) if (~mask).any() else float("nan")
    return {
        "scores": scores.detach().cpu(),
        "hit_mean": hit_mean,
        "miss_mean": miss_mean,
        "top_hit_rank": top_hit_rank(scores, mask),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--params", required=True,
                   help="path to a .pt file saved by 02_train.py")
    p.add_argument("--proteins", nargs="+", default=None,
                   help="protein IDs (default: proteins recorded in the .pt)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device}  dtype={dtype}")

    params_path = Path(args.params)
    if not params_path.exists():
        raise FileNotFoundError(
            f"{params_path} not found. Run 02_train.py first."
        )
    ckpt = torch.load(params_path, map_location="cpu", weights_only=False)
    proteins = args.proteins or ckpt.get("proteins") or []
    if not proteins:
        raise SystemExit("No --proteins given and none recorded in the .pt file.")
    print(f"params={params_path}  proteins={proteins}")

    before = default_params(device, dtype)
    after = trained_params(params_path, device, dtype)

    all_before = []
    all_after = []
    all_hit = []
    print(f"\n{'protein':<8} {'phase':<7} {'hit μ':>10} {'miss μ':>10} {'Δhit':>10} {'Δmiss':>10} {'rank*':>7}")
    for pid in proteins:
        lp = load_protein(pid, device=device, dtype=dtype)
        b = evaluate_one(lp, before)
        a = evaluate_one(lp, after)
        dh = a["hit_mean"] - b["hit_mean"]
        dm = a["miss_mean"] - b["miss_mean"]
        print(f"{pid:<8} {'before':<7} {b['hit_mean']:>10.3f} {b['miss_mean']:>10.3f} {'':>10} {'':>10} "
              f"{b['top_hit_rank'] if b['top_hit_rank'] is not None else '-':>7}")
        print(f"{'':<8} {'after':<7} {a['hit_mean']:>10.3f} {a['miss_mean']:>10.3f} {dh:>10.3f} {dm:>10.3f} "
              f"{a['top_hit_rank'] if a['top_hit_rank'] is not None else '-':>7}")
        all_before.append(b["scores"].numpy())
        all_after.append(a["scores"].numpy())
        all_hit.append(lp.inputs.hit_mask.detach().cpu().numpy())

    print("\n* rank = best (1-based) rank of any Hit pose in the F sorted poses")

    before_flat = np.concatenate(all_before)
    after_flat = np.concatenate(all_after)
    hit_flat = np.concatenate(all_hit).astype(bool)

    png = OUT_DIR / "03_before_after.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(before_flat[~hit_flat], after_flat[~hit_flat],
               s=18, c="lightgrey", label="Miss", edgecolor="none")
    ax.scatter(before_flat[hit_flat], after_flat[hit_flat],
               s=30, c="tab:blue", label="Hit", edgecolor="k", linewidth=0.5)
    lo = float(min(before_flat.min(), after_flat.min()))
    hi = float(max(before_flat.max(), after_flat.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("score (before: default params)")
    ax.set_ylabel("score (after: trained params)")
    ax.set_title(f"before vs after — {', '.join(proteins)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
