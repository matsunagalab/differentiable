"""End-to-end FFT docking on 1KXQ — pure-PyTorch, no C/Fortran ZDOCK.

Demonstrates `zdock.search.docking_search`: for each of N sampled
rotations of the ligand, evaluate the full `docking_score_elec` score
at every translation cell in one batched FFT, then return the top-K
(rotation, translation) poses.

What this script does:

  1. Load 1KXQ receptor + reference ligand from the phase-5 h5 refs.
  2. Generate a small rotation grid (default 64 random SO(3) samples).
  3. Run `docking_search` with Julia-default learnable parameters
     (α=0.01, β=3.0, iface and charge from the default LUTs).
  4. Report the top-K (score, quaternion_idx, translation) tuples.
  5. Cross-check every reported pose by plugging the
     rotation+translation back into `docking_score_elec`, confirming
     bit-exact agreement (guarantees the FFT path is sound).

Runtime (float64, 1KXQ, 117×115×129 grid):

    * CPU: ~5 s receptor precompute + ~3 s per rotation
    * CUDA: ~1 s precompute + ~50 ms per rotation
    * 64 rotations: ~3.5 min on CPU, ~5 s on CUDA

Use `--n-rotations` / `--spacing` to scale up. A 6° ZDOCK-compatible
rotation grid would be 54000 rotations; generating all of them with
this demo is a ~hours-scale CPU run / ~10 min on CUDA.

Usage:

    uv run python examples/04_fft_docking.py                    # default 64 rotations
    uv run python examples/04_fft_docking.py --n-rotations 256
    CUDA_VISIBLE_DEVICES=0 uv run python examples/04_fft_docking.py --device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from zdock.atomtypes import charge_score as default_charge_score_lut
from zdock.atomtypes import iface_ij as default_iface_ij
from zdock.score import docking_score_elec
from zdock.rotation_grid import random_quaternions
from zdock.search import _rotate_batch, docking_search

from _data import load_protein, resolve_device, default_dtype

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-rotations", type=int, default=64,
                    help="number of random SO(3) rotations to search "
                         "(default 64 — small for a quick demo)")
    ap.add_argument("--ntop", type=int, default=10,
                    help="number of top (q, t) poses to report (default 10)")
    ap.add_argument("--spacing", type=float, default=1.2,
                    help="FFT grid spacing in Å (default 1.2)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the quaternion sampler")
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda | mps (default auto)")
    ap.add_argument("--rot-chunk-size", type=int, default=16,
                    help="rotations processed per FFT batch (VRAM knob)")
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "04_fft_docking_top.txt",
                    help="where to write the top-K pose table")
    args = ap.parse_args()

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device}  dtype={dtype}  n_rotations={args.n_rotations}  "
          f"ntop={args.ntop}")

    # ------------------------------------------------------------------
    # 1. Load 1KXQ and the reference ligand (orient/decentered).
    # ------------------------------------------------------------------
    print("loading 1KXQ phase-5 refs...")
    loaded = load_protein("1KXQ", device=device, dtype=dtype)
    rec = loaded.inputs
    raw = loaded.raw

    # The reference ligand (orient+decentered) is stored as (3, N_lig) in
    # the Julia h5; load_protein only pulls the pose array lig_xyz. We
    # need lig_xyz_for_grid for the docking search.
    import numpy as np
    lig_xyz_ref = torch.as_tensor(
        np.asarray(raw["lig_xyz_for_grid"]).T, dtype=dtype, device=device,
    ).contiguous()
    print(f"  rec atoms: {rec.rec_xyz.shape[0]}  "
          f"lig atoms: {lig_xyz_ref.shape[0]}")

    # Julia-default parameters (LUT iface, LUT charge).
    alpha = torch.tensor(0.01, dtype=dtype, device=device)
    beta = torch.tensor(3.0, dtype=dtype, device=device)
    iface_flat = default_iface_ij(
        device=device, dtype=dtype, flat=True,
    ).clone()
    charge_lut = default_charge_score_lut(
        device=device, dtype=dtype,
    ).clone()

    # ------------------------------------------------------------------
    # 2. Rotation grid — small random SO(3) sample for the demo.
    # ------------------------------------------------------------------
    quaternions = random_quaternions(
        args.n_rotations, seed=args.seed, device=device, dtype=dtype,
    )
    # Seed rotation 0 with the identity so the top-K list always
    # contains the no-rotation baseline for comparison.
    quaternions = torch.cat([
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=dtype, device=device),
        quaternions,
    ], dim=0)
    n_rot = quaternions.shape[0]
    print(f"  rotation grid: {n_rot} (identity + {args.n_rotations} random)")

    # ------------------------------------------------------------------
    # 3. Run FFT search.
    # ------------------------------------------------------------------
    print("running docking_search ...")
    t0 = time.time()
    result = docking_search(
        rec.rec_xyz, rec.rec_radius, rec.rec_sasa,
        rec.rec_atomtype_id, rec.rec_charge_id,
        lig_xyz_ref, rec.lig_radius, rec.lig_sasa,
        rec.lig_atomtype_id, rec.lig_charge_id,
        quaternions=quaternions,
        alpha=alpha, iface_ij_flat=iface_flat, beta=beta,
        charge_score_lut=charge_lut,
        spacing=args.spacing, ntop=args.ntop,
        rot_chunk_size=args.rot_chunk_size,
    )
    elapsed = time.time() - t0
    per_rot_ms = 1000 * elapsed / n_rot
    print(f"  done in {elapsed:.1f} s  ({per_rot_ms:.0f} ms / rotation)")

    # ------------------------------------------------------------------
    # 4. Report top-K poses.
    # ------------------------------------------------------------------
    print("")
    header = (f"{'rank':>4}  {'score':>14}  {'q_idx':>5}  "
              f"{'tx':>8}  {'ty':>8}  {'tz':>8}")
    print(header)
    print("-" * len(header))
    for i in range(args.ntop):
        s = result.scores[i].item()
        qi = int(result.quat_indices[i].item())
        t = result.translations[i].tolist()
        print(f"{i+1:>4}  {s:>+14.4e}  {qi:>5d}  "
              f"{t[0]:>+8.3f}  {t[1]:>+8.3f}  {t[2]:>+8.3f}")

    # ------------------------------------------------------------------
    # 5. Cross-check every top-K pose against docking_score_elec.
    #    Ensures the FFT path is sound end-to-end (any discrepancy here
    #    would indicate a regression in search.py).
    # ------------------------------------------------------------------
    print("\nverifying each pose via docking_score_elec ...")
    max_rel = 0.0
    for i in range(args.ntop):
        qi = int(result.quat_indices[i].item())
        t_ang = result.translations[i]
        q = quaternions[qi]
        lig_rot = _rotate_batch(lig_xyz_ref, q.unsqueeze(0))[0]
        lig_pose = (lig_rot + t_ang).unsqueeze(0)
        tot_e = docking_score_elec(
            rec.rec_xyz, rec.rec_radius, rec.rec_sasa,
            rec.rec_atomtype_id, rec.rec_charge_id,
            lig_pose, rec.lig_radius, rec.lig_sasa,
            rec.lig_atomtype_id, rec.lig_charge_id,
            alpha=alpha, iface_ij_flat=iface_flat, beta=beta,
            charge_score=charge_lut,
            lig_xyz_for_grid=lig_xyz_ref, spacing=args.spacing,
        ).item()
        tot_fft = result.scores[i].item()
        rel = abs(tot_e - tot_fft) / (abs(tot_e) + 1.0)
        max_rel = max(max_rel, rel)

    tol_ok = max_rel < (1e-4 if dtype == torch.float32 else 1e-10)
    print(f"  max relative diff: {max_rel:.2e}  "
          f"{'OK' if tol_ok else 'UNEXPECTED'}")

    # ------------------------------------------------------------------
    # 6. Save top-K to disk.
    # ------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(f"# FFT docking search on 1KXQ\n")
        f.write(f"# device={device} dtype={dtype} spacing={args.spacing}\n")
        f.write(f"# n_rotations={n_rot} ntop={args.ntop}\n")
        f.write(f"# runtime={elapsed:.1f}s ({per_rot_ms:.0f}ms/rot)\n")
        f.write(f"# max_rel_diff_vs_docking_score_elec={max_rel:.2e}\n")
        f.write(f"{header}\n")
        for i in range(args.ntop):
            s = result.scores[i].item()
            qi = int(result.quat_indices[i].item())
            t = result.translations[i].tolist()
            f.write(f"{i+1:>4}  {s:>+14.4e}  {qi:>5d}  "
                    f"{t[0]:>+8.3f}  {t[1]:>+8.3f}  {t[2]:>+8.3f}\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
