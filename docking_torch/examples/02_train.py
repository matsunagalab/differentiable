"""Minimal training demo.

Runs `zdock.train.train` on one or more proteins and saves:

    out/trained_params.pt   — dict with alpha, beta, iface, charge, history, proteins
    out/02_loss_curve.png   — epoch vs loss plot

To extend the training set, add entries to `DATASETS` in
`examples/_data.py` (one line per protein's HDF5 ref). This script
needs no changes — just pass the new IDs on `--proteins`.

Run:
    uv run python examples/02_train.py
    uv run python examples/02_train.py --proteins 1KXQ 1F51 2VDB --epochs 200
    uv run python examples/02_train.py --device cuda --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from zdock.train import train

from _data import DEFAULT_TRAIN_SET, default_dtype, load_protein, resolve_device

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def plot_loss(path: Path, history: list[float], proteins: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, lw=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(f"training loss — proteins: {', '.join(proteins)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proteins", nargs="+", default=DEFAULT_TRAIN_SET,
                   help=f"protein IDs (default: {DEFAULT_TRAIN_SET})")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--out", default=str(OUT_DIR / "trained_params.pt"))
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device}  dtype={dtype}  proteins={args.proteins}  "
          f"epochs={args.epochs}  lr={args.lr}")

    proteins = [load_protein(pid, device=device, dtype=dtype).inputs
                for pid in args.proteins]
    for pid, pi in zip(args.proteins, proteins):
        n_hit = int(pi.hit_mask.sum().item())
        print(f"  {pid}: F={pi.lig_xyz.shape[0]} poses, "
              f"{n_hit} Hit / {pi.lig_xyz.shape[0] - n_hit} Miss")

    out = train(
        proteins,
        n_epoch=args.epochs,
        lr=args.lr,
        device=device,
        dtype=dtype,
        progress_every=args.progress_every,
    )

    hist = out["history"]["loss"]
    print(f"\ninitial loss = {hist[0]:.4e}")
    print(f"final   loss = {hist[-1]:.4e}  "
          f"(reduction {(1 - hist[-1] / hist[0]) * 100:.1f}%)")
    print(f"α = {out['alpha'].item():.4e}  β = {out['beta'].item():.4e}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "alpha": out["alpha"].cpu(),
            "beta": out["beta"].cpu(),
            "iface": out["iface"].cpu(),
            "charge": out["charge"].cpu(),
            "history": out["history"],
            "proteins": list(args.proteins),
            "epochs": args.epochs,
            "lr": args.lr,
        },
        out_path,
    )
    print(f"\nwrote {out_path}")

    png = OUT_DIR / "02_loss_curve.png"
    plot_loss(png, hist, args.proteins)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
