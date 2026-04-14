"""Minimal docking-style scoring + visualization demo.

Loads the 1KXQ top-10 ZDOCK candidate poses from the phase5 reference,
scores them with the default parameters (α=0.01, β=3.0, LUT iface /
charge), prints a ranked table, and writes:

    out/01_poses.pdb   — multi-model point-cloud (receptor + top-5 lig)
    out/01_poses.png   — matplotlib 3D scatter preview

Run:
    uv run python examples/01_score_and_visualize.py
    uv run python examples/01_score_and_visualize.py --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no GUI backend needed
import matplotlib.pyplot as plt
import torch

from zdock.atomtypes import charge_score as charge_score_lut
from zdock.atomtypes import iface_ij

from _data import default_dtype, load_protein, resolve_device

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def write_multimodel_pdb(
    path: Path,
    rec_xyz: torch.Tensor,
    lig_xyzs: list[torch.Tensor],
) -> None:
    """Dump a point-cloud PDB: receptor as model 1 chain R, then one
    MODEL per ligand pose (chain L). Resname XXX so PyMOL treats atoms
    as generic points; element C so colouring/selection is trivial."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def _atom(i: int, chain: str, x: float, y: float, z: float) -> str:
        # Columns strictly per PDB spec so PyMOL/ChimeraX parse cleanly.
        return (
            f"ATOM  {i:>5d}  CA  XXX {chain}{(i % 9999) + 1:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00           C  \n"
        )

    with path.open("w") as f:
        f.write("MODEL     1\n")
        for i, (x, y, z) in enumerate(rec_xyz.detach().cpu().numpy(), start=1):
            f.write(_atom(i, "R", float(x), float(y), float(z)))
        f.write("ENDMDL\n")
        for m, lig in enumerate(lig_xyzs, start=2):
            f.write(f"MODEL {m:>5d}\n")
            for i, (x, y, z) in enumerate(lig.detach().cpu().numpy(), start=1):
                f.write(_atom(i, "L", float(x), float(y), float(z)))
            f.write("ENDMDL\n")
        f.write("END\n")


def plot_scatter(
    path: Path,
    rec_xyz: torch.Tensor,
    lig_xyzs: list[torch.Tensor],
    scores: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")

    r = rec_xyz.detach().cpu().numpy()
    ax.scatter(r[:, 0], r[:, 1], r[:, 2], c="lightgrey", s=2, alpha=0.3, label="receptor")

    cmap = plt.get_cmap("viridis")
    vmin, vmax = min(scores), max(scores)
    for lig, sc in zip(lig_xyzs, scores):
        l = lig.detach().cpu().numpy()
        color = cmap((sc - vmin) / (vmax - vmin + 1e-12))
        ax.scatter(l[:, 0], l[:, 1], l[:, 2], color=color, s=3,
                   label=f"score={sc:.1f}")

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_title("1KXQ — receptor (grey) + top poses (viridis by score)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--topk", type=int, default=5, help="poses to export to PDB/PNG")
    p.add_argument("--protein", default="1KXQ")
    args = p.parse_args()

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device}  dtype={dtype}")

    lp = load_protein(args.protein, device=device, dtype=dtype)
    pi = lp.inputs
    alpha = torch.tensor(0.01, device=device, dtype=dtype)
    beta = torch.tensor(3.0, device=device, dtype=dtype)
    iface = iface_ij(device=device, dtype=dtype, flat=True)
    charge = charge_score_lut(device=device, dtype=dtype)

    with torch.no_grad():
        scores = pi.call(alpha, iface, beta, charge)
    scores_np = scores.detach().cpu().numpy()

    # Ground-truth (Julia's score_coulomb_total, stored in the HDF5) for
    # a sanity-check side column — validates the Python port is in sync.
    ref_scores = lp.raw.get("score_coulomb_total")

    print(f"\n{args.protein}: {pi.lig_xyz.shape[0]} poses scored")
    print(f"{'rank':>4} {'pose':>4} {'score_python':>14} {'score_julia_ref':>16}")
    order = sorted(range(len(scores_np)), key=lambda i: -scores_np[i])
    for rank, idx in enumerate(order, start=1):
        ref = float(ref_scores[idx]) if ref_scores is not None else float("nan")
        print(f"{rank:>4d} {idx:>4d} {scores_np[idx]:>14.3f} {ref:>16.3f}")

    top_indices = order[: args.topk]
    lig_poses = [pi.lig_xyz[i] for i in top_indices]
    top_scores = [float(scores_np[i]) for i in top_indices]

    pdb_path = OUT_DIR / "01_poses.pdb"
    png_path = OUT_DIR / "01_poses.png"
    write_multimodel_pdb(pdb_path, pi.rec_xyz, lig_poses)
    plot_scatter(png_path, pi.rec_xyz, lig_poses, top_scores)
    print(f"\nwrote {pdb_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
