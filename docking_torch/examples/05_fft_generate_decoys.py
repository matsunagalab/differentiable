"""Multi-protein FFT decoy generation over BM4.

Iterate over every protein in `bm4_full.h5`, run `docking_search` with
a configurable rotation grid, and save the top-N (score, rotation,
translation, reconstructed-ligand-coords) tuples to an output h5.
This is the pure-PyTorch replacement for ZDOCK's decoy-generation
step — no external C/Fortran binary.

Output h5 layout (one group per protein):

    /<PROTEIN>/
        lig_xyz_decoy      (F, N_lig, 3) float32   reconstructed poses
                                                  in the receptor's raw
                                                  (pre-decenter) frame
        score              (F,)          float32   our scoring function
        rotation_quat      (F, 4)        float32   quaternion index into
                                                  the (F,) set used
        translation        (F, 3)        float32   cartesian Å in the
                                                  decentered search frame
        rmsd_vs_bm4_best   (F,)          float32   optional — RMSD to the
                                                  BM4 decoy with smallest
                                                  native RMSD (a proxy
                                                  for RMSD-to-crystal)

Root attrs record the rotation grid type, sampling size, ntop, device,
scorer params, and runtime.

Examples:
    # Default: 4096 random rotations, top-2000 decoys, Julia-default params
    CUDA_VISIBLE_DEVICES=0 uv run python examples/05_fft_generate_decoys.py \
        --device cuda --out out/bm4_fft_decoys.h5

    # Subset for quick iteration + trained params
    uv run python examples/05_fft_generate_decoys.py --device cuda \
        --proteins 1PPE 1KXQ 2SIC --n-rotations 2048 \
        --params-ckpt out/trained_params_rank.pt

    # Euler 6° grid (ZDOCK-comparable coverage, full ~54k rotations)
    uv run python examples/05_fft_generate_decoys.py --device cuda \
        --euler-deg 6.0 --ntop 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from zdock.atomtypes import charge_score as default_charge_score_lut
from zdock.atomtypes import iface_ij as default_iface_ij
from zdock.data import list_proteins
from zdock.rotation_grid import euler_quaternions, random_quaternions
from zdock.search import (
    _rotate_batch,
    docking_search,
    prepare_ligand,
)

DEFAULT_H5 = Path(__file__).resolve().parent.parent / "datasets" / "bm4_full.h5"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "out" / "bm4_fft_decoys.h5"


def resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def default_dtype(device: torch.device) -> torch.dtype:
    return torch.float64 if device.type == "cpu" else torch.float32


def build_quaternions(
    args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Either a random or Euler grid per the CLI."""
    if args.euler_deg is not None:
        q = euler_quaternions(deg=args.euler_deg, device=device, dtype=dtype)
    else:
        q = random_quaternions(
            args.n_rotations, seed=args.seed, device=device, dtype=dtype,
        )
    return q


def load_params(
    ckpt_path: Path | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (alpha, iface_flat, beta, charge_lut). If ``ckpt_path`` is
    given, load from a trained checkpoint; else Julia defaults."""
    if ckpt_path is None:
        return (
            torch.tensor(0.01, dtype=dtype, device=device),
            default_iface_ij(device=device, dtype=dtype, flat=True).clone(),
            torch.tensor(3.0, dtype=dtype, device=device),
            default_charge_score_lut(device=device, dtype=dtype).clone(),
        )
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    # `02_train.py` saves keys "alpha", "iface", "charge" (β is frozen at 3.0)
    return (
        ckpt["alpha"].to(device=device, dtype=dtype),
        ckpt["iface"].to(device=device, dtype=dtype),
        torch.tensor(3.0, dtype=dtype, device=device),
        ckpt["charge"].to(device=device, dtype=dtype),
    )


def run_one_protein(
    protein_id: str,
    h5_in: h5py.File,
    *,
    quaternions: torch.Tensor,
    alpha: torch.Tensor,
    iface_flat: torch.Tensor,
    beta: torch.Tensor,
    charge_lut: torch.Tensor,
    spacing: float,
    ntop: int,
    rot_chunk_size: int,
    device: torch.device,
    dtype: torch.dtype,
    compute_rmsd: bool,
) -> dict | None:
    """Run docking_search on one protein. Returns None on OOM after
    retry cascade."""
    g = h5_in[protein_id]

    rec_xyz_raw = torch.as_tensor(g["rec_xyz"][()], dtype=dtype, device=device)
    rec_radius = torch.as_tensor(g["rec_radius"][()], dtype=dtype, device=device)
    rec_sasa = torch.as_tensor(g["rec_sasa"][()], dtype=dtype, device=device)
    rec_aid = torch.as_tensor(
        g["rec_atomtype_id"][()], dtype=torch.int64, device=device,
    )
    rec_cid = torch.as_tensor(
        g["rec_charge_id"][()], dtype=torch.int64, device=device,
    )
    lig_all = torch.as_tensor(g["lig_xyz"][()], dtype=dtype, device=device)
    lig_radius = torch.as_tensor(g["lig_radius"][()], dtype=dtype, device=device)
    lig_sasa = torch.as_tensor(g["lig_sasa"][()], dtype=dtype, device=device)
    lig_aid = torch.as_tensor(
        g["lig_atomtype_id"][()], dtype=torch.int64, device=device,
    )
    lig_cid = torch.as_tensor(
        g["lig_charge_id"][()], dtype=torch.int64, device=device,
    )
    rmsd_bm4 = (
        torch.as_tensor(g["rmsd"][()], dtype=dtype, device=device)
        if "rmsd" in g else None
    )

    rec_com = rec_xyz_raw.mean(dim=0)
    rec_xyz = rec_xyz_raw - rec_com
    lig_xyz_ref = prepare_ligand(lig_all[0], lig_aid)

    # Retry-on-OOM cascade.
    chunk = rot_chunk_size
    for attempt in range(4):
        try:
            result = docking_search(
                rec_xyz, rec_radius, rec_sasa, rec_aid, rec_cid,
                lig_xyz_ref, lig_radius, lig_sasa, lig_aid, lig_cid,
                quaternions=quaternions,
                alpha=alpha, iface_ij_flat=iface_flat, beta=beta,
                charge_score_lut=charge_lut,
                spacing=spacing, ntop=ntop, rot_chunk_size=chunk,
            )
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk = max(chunk // 2, 1)
            if chunk == 1 and attempt == 3:
                return None
    else:
        return None

    # Reconstruct each top pose's ligand coords in the RAW frame (same as
    # BM4 `lig_xyz`). `prepare_ligand` orient+decentered the reference; to
    # get back to raw, rotate, translate (search frame), then add rec_com.
    F = result.scores.shape[0]
    N_lig = lig_xyz_ref.shape[0]
    lig_xyz_decoy = torch.empty((F, N_lig, 3), dtype=dtype, device=device)
    # Batch the rotation (one (F, N_lig, 3) tensor).
    q_per_pose = quaternions[result.quat_indices]   # (F, 4)
    lig_rot_all = _rotate_batch(lig_xyz_ref, q_per_pose)  # (F, N_lig, 3)
    lig_xyz_decoy = lig_rot_all + result.translations.unsqueeze(-2)
    lig_xyz_decoy = lig_xyz_decoy + rec_com  # back to raw frame

    out = {
        "lig_xyz_decoy": lig_xyz_decoy.cpu().to(torch.float32).numpy(),
        "score": result.scores.cpu().to(torch.float32).numpy(),
        "rotation_quat": q_per_pose.cpu().to(torch.float32).numpy(),
        "translation": result.translations.cpu().to(torch.float32).numpy(),
    }

    if compute_rmsd and rmsd_bm4 is not None:
        # "Pseudo-ground-truth" = BM4 decoy with smallest stored RMSD.
        nn_idx = int(rmsd_bm4.argmin().item())
        nn_target = lig_all[nn_idx]  # (N_lig, 3) in raw frame
        rmsd_vs_nn = (
            (lig_xyz_decoy - nn_target)
            .pow(2).sum(-1).mean(-1).sqrt()
        )
        out["rmsd_vs_bm4_best"] = rmsd_vs_nn.cpu().to(torch.float32).numpy()
        out["_bm4_best_decoy_idx"] = nn_idx
        out["_bm4_best_decoy_rmsd_to_crystal"] = float(rmsd_bm4[nn_idx].item())

    out["_final_chunk"] = chunk
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5,
                    help=f"input BM4 h5 (default {DEFAULT_H5})")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output h5 (default {DEFAULT_OUT})")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--proteins", nargs="+", default=None,
                    help="subset of protein IDs to process (default: all)")
    rot_g = ap.add_mutually_exclusive_group()
    rot_g.add_argument("--n-rotations", type=int, default=4096,
                       help="number of random SO(3) rotations (default 4096)")
    rot_g.add_argument("--euler-deg", type=float, default=None,
                       help="if set, use Euler-angle grid at this step "
                            "instead of random sampling (e.g. 6.0 for "
                            "ZDOCK-comparable ~54k rotations)")
    ap.add_argument("--seed", type=int, default=42,
                    help="seed for random quaternion sampler")
    ap.add_argument("--ntop", type=int, default=2000,
                    help="top-K poses retained per protein (default 2000)")
    ap.add_argument("--spacing", type=float, default=1.2,
                    help="FFT grid spacing in Å (default 1.2)")
    ap.add_argument("--rot-chunk-size", type=int, default=64,
                    help="rotations per FFT batch; auto-halves on OOM "
                         "(default 64)")
    ap.add_argument("--params-ckpt", type=Path, default=None,
                    help="load trained α/iface/charge from this 02_train "
                         "checkpoint; default = Julia LUT defaults")
    ap.add_argument("--no-rmsd", action="store_true",
                    help="skip RMSD-to-near-native computation")
    args = ap.parse_args()

    if not args.h5.exists():
        raise SystemExit(f"input h5 not found: {args.h5}")

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device}  dtype={dtype}")

    quaternions = build_quaternions(args, device=device, dtype=dtype)
    rot_desc = (
        f"random n={quaternions.shape[0]} (seed={args.seed})"
        if args.euler_deg is None else
        f"euler deg={args.euler_deg} n={quaternions.shape[0]}"
    )
    print(f"rotation grid: {rot_desc}")

    alpha, iface_flat, beta, charge_lut = load_params(
        args.params_ckpt, device=device, dtype=dtype,
    )
    print(f"scorer params: alpha={alpha.item():.4e}  beta={beta.item():.4e}  "
          f"{'(trained ckpt)' if args.params_ckpt else '(Julia defaults)'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_names = list_proteins(args.h5)
    names = args.proteins if args.proteins else all_names
    missing = [p for p in names if p not in all_names]
    if missing:
        raise SystemExit(f"proteins not in {args.h5}: {missing}")
    print(f"proteins to process: {len(names)}")

    total_start = time.time()

    # Open input once; open output in append mode (skip proteins already written).
    with h5py.File(args.h5, "r") as h5_in, h5py.File(args.out, "a") as h5_out:
        # Root attrs (overwrite each run so they reflect the last config).
        h5_out.attrs["rotation_desc"] = rot_desc
        h5_out.attrs["n_rotations"] = int(quaternions.shape[0])
        h5_out.attrs["ntop"] = int(args.ntop)
        h5_out.attrs["spacing"] = float(args.spacing)
        h5_out.attrs["alpha"] = float(alpha.item())
        h5_out.attrs["beta"] = float(beta.item())
        if args.params_ckpt is not None:
            h5_out.attrs["params_ckpt"] = str(args.params_ckpt)

        for i, prot in enumerate(names, 1):
            if prot in h5_out and "lig_xyz_decoy" in h5_out[prot]:
                print(f"[{i:>3}/{len(names)}] {prot}: already done, skip")
                continue

            t0 = time.time()
            result = run_one_protein(
                prot, h5_in,
                quaternions=quaternions,
                alpha=alpha, iface_flat=iface_flat, beta=beta,
                charge_lut=charge_lut,
                spacing=args.spacing, ntop=args.ntop,
                rot_chunk_size=args.rot_chunk_size,
                device=device, dtype=dtype,
                compute_rmsd=not args.no_rmsd,
            )
            dt = time.time() - t0
            if result is None:
                print(f"[{i:>3}/{len(names)}] {prot}: OOM after retry cascade, skip")
                continue

            best_rmsd_str = ""
            if "rmsd_vs_bm4_best" in result:
                best_rmsd = float(np.min(result["rmsd_vs_bm4_best"]))
                best_rmsd_str = (
                    f"  best_rmsd_to_near_native={best_rmsd:.2f}Å"
                )
            chunk = result.pop("_final_chunk")
            print(
                f"[{i:>3}/{len(names)}] {prot}: {dt:>5.1f}s  chunk={chunk}  "
                f"top-score={float(np.max(result['score'])):.2e}{best_rmsd_str}"
            )

            # Write to h5.
            if prot in h5_out:
                del h5_out[prot]
            grp = h5_out.create_group(prot)
            for k, v in result.items():
                if k.startswith("_"):
                    grp.attrs[k[1:]] = v
                else:
                    grp.create_dataset(
                        k, data=v, compression="gzip", compression_opts=3,
                    )
            h5_out.flush()

    total = time.time() - total_start
    print(f"\nDone in {total:.1f}s  ({total/60:.1f} min)  → {args.out}")


if __name__ == "__main__":
    main()
