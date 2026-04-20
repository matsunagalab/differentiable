"""Build a consolidated multi-protein training h5 from BM4 inputs.

Reads the ZDOCK Benchmark4 decoy directory (`input_pdbs/` + `results/`) and
writes a single h5 with one group per protein. Each group contains the
rec/lig prepared inputs plus the full 54000-pose RMSD vector and a Hit/Miss
bool mask (`hit_mask = rmsd <= threshold`).

Usage (see `--help` for all options):

    uv run python scripts/build_training_dataset.py \\
        --benchmark-root ../docking/decoys_bm4_zd3.0.2_6deg_fixed \\
        --output datasets/bm4_full.h5

The operation is idempotent: groups that already exist in the output file
are skipped unless `--overwrite` is set.
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# `src` is added to the path so the script can run in-place without
# `pip install -e .`. (uv run handles this automatically anyway.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zdock.atomtypes import set_atomtype_id, set_charge, set_radius  # noqa: E402
from zdock.io import parse_pdb_ms  # noqa: E402
from zdock.sasa import compute_sasa  # noqa: E402
from zdock.zdock_output import generate_lig_coords, parse_out_file  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _discover_proteins(benchmark_root: Path) -> list[str]:
    """List proteins whose rec+lig PDB, .out, and .rmsds are all present."""
    in_pdbs = benchmark_root / "input_pdbs"
    results = benchmark_root / "results"
    names: set[str] = set()
    for p in in_pdbs.glob("*_r_u.pdb.ms"):
        n = p.name.removesuffix("_r_u.pdb.ms")
        if (
            (in_pdbs / f"{n}_l_u.pdb.ms").exists()
            and (results / f"{n}.zd3.0.2.fg.fixed.out").exists()
            and (results / f"{n}.zd3.0.2.fg.fixed.out.rmsds").exists()
        ):
            names.add(n)
    return sorted(names)


def _process_protein(
    name: str,
    benchmark_root: Path,
    threshold: float,
    max_poses: int | None,
    sasa_device: torch.device,
) -> dict[str, np.ndarray]:
    """Compute all tensors for one protein as float32/int64 numpy arrays.

    SASA runs on `sasa_device` (CPU by default; MPS/CUDA override available
    via the env var `ZDOCK_DEVICE`). Everything else is numpy — the
    rotation+translation on the pose batch is a single big matmul and stays
    fast on CPU with numpy/torch-CPU.
    """
    rec_pdb = benchmark_root / "input_pdbs" / f"{name}_r_u.pdb.ms"
    lig_pdb = benchmark_root / "input_pdbs" / f"{name}_l_u.pdb.ms"
    out_file = benchmark_root / "results" / f"{name}.zd3.0.2.fg.fixed.out"
    rmsd_file = benchmark_root / "results" / f"{name}.zd3.0.2.fg.fixed.out.rmsds"

    rec = parse_pdb_ms(rec_pdb)
    lig = parse_pdb_ms(lig_pdb)

    # Atom-type / charge / radius, all from name-based ports.
    rec_atomtype_id = set_atomtype_id(rec.resname, rec.atomname).numpy().astype(np.int64)
    rec_charge_id = set_charge(rec.resname, rec.atomname).numpy().astype(np.int64)
    rec_radius = set_radius(rec.atomname).numpy().astype(np.float32)

    lig_atomtype_id = set_atomtype_id(lig.resname, lig.atomname).numpy().astype(np.int64)
    lig_charge_id = set_charge(lig.resname, lig.atomname).numpy().astype(np.int64)
    lig_radius = set_radius(lig.atomname).numpy().astype(np.float32)

    # SASA — runs on the configured device for speed, result back to numpy.
    rec_sasa = compute_sasa(
        torch.as_tensor(rec.xyz, dtype=torch.float64, device=sasa_device),
        torch.as_tensor(rec_radius, dtype=torch.float64, device=sasa_device),
    ).cpu().numpy().astype(np.float32)
    lig_sasa = compute_sasa(
        torch.as_tensor(lig.xyz, dtype=torch.float64, device=sasa_device),
        torch.as_tensor(lig_radius, dtype=torch.float64, device=sasa_device),
    ).cpu().numpy().astype(np.float32)

    # Pose generation (54000 poses, vectorized).
    zd = parse_out_file(out_file)
    lig_xyz_poses = generate_lig_coords(
        torch.as_tensor(lig.xyz, dtype=torch.float64),
        zd,
        n_poses=max_poses,
        dtype=torch.float64,
    ).numpy().astype(np.float32)

    # RMSD file: 54000 lines of `<pose_id>\t<rmsd>` (some proteins add a
    # third flag column — interpretation varies by BM4 version; we only
    # need column 1, the RMSD in Å).
    rmsd_raw = np.loadtxt(rmsd_file, dtype=np.float64)
    if rmsd_raw.ndim != 2 or rmsd_raw.shape[1] < 2:
        raise ValueError(
            f"{rmsd_file}: expected (N, >=2) layout, got shape {rmsd_raw.shape}"
        )
    rmsd = rmsd_raw[:, 1].astype(np.float32)
    if rmsd.shape[0] != zd.pose_rot.shape[0]:
        raise ValueError(
            f"{name}: .out has {zd.pose_rot.shape[0]} poses but .rmsds has "
            f"{rmsd.shape[0]} rows"
        )
    if max_poses is not None:
        rmsd = rmsd[:max_poses]

    hit_mask = (rmsd <= threshold)

    return {
        "rec_xyz": rec.xyz.astype(np.float32),
        "rec_radius": rec_radius,
        "rec_sasa": rec_sasa,
        "rec_atomtype_id": rec_atomtype_id,
        "rec_charge_id": rec_charge_id,
        "lig_xyz": lig_xyz_poses,
        "lig_radius": lig_radius,
        "lig_sasa": lig_sasa,
        "lig_atomtype_id": lig_atomtype_id,
        "lig_charge_id": lig_charge_id,
        "rmsd": rmsd,
        "hit_mask": hit_mask,
    }


def _write_protein(
    f: h5py.File,
    name: str,
    data: dict[str, np.ndarray],
    *,
    compression: str | None,
    overwrite: bool,
):
    if name in f:
        if not overwrite:
            raise RuntimeError(
                f"group {name!r} already exists (use --overwrite to replace)"
            )
        del f[name]
    g = f.create_group(name)
    # `lig_xyz` is the big one; chunk per-pose so random-access reads of
    # a handful of poses don't decompress the entire trajectory.
    chunk = None
    if compression is not None:
        F, N_lig, _ = data["lig_xyz"].shape
        chunk = (1, N_lig, 3)

    for key, arr in data.items():
        kwargs: dict = {}
        if compression is not None:
            kwargs["compression"] = compression
            if key == "lig_xyz":
                kwargs["chunks"] = chunk
        g.create_dataset(key, data=arr, **kwargs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--benchmark-root", type=Path, required=True,
        help="path to decoys_bm4_zd3.0.2_6deg_fixed/",
    )
    ap.add_argument(
        "--output", type=Path, required=True,
        help="output h5 path (parent dir created if missing)",
    )
    ap.add_argument(
        "--proteins", nargs="*", default=None,
        help="optional whitelist. Default: all proteins found under benchmark-root.",
    )
    ap.add_argument(
        "--max-poses", type=int, default=None,
        help="optional cap on poses per protein (default: all 54000).",
    )
    ap.add_argument(
        "--threshold", type=float, default=2.5,
        help="RMSD threshold in Å for hit_mask (default: 2.5, thesis convention).",
    )
    ap.add_argument(
        "--compression", choices=["gzip", "none"], default="gzip",
        help="h5 dataset compression (default: gzip).",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="overwrite existing groups in the output file.",
    )
    ap.add_argument(
        "--skip-errors", action="store_true",
        help="skip proteins that raise during prep (default: fail fast).",
    )
    args = ap.parse_args(argv)

    if not args.benchmark_root.exists():
        ap.error(f"--benchmark-root does not exist: {args.benchmark_root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device_name = os.environ.get("ZDOCK_DEVICE", "cpu").lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    if device_name == "mps" and not torch.backends.mps.is_available():
        device_name = "cpu"
    sasa_device = torch.device(device_name)

    names = args.proteins if args.proteins else _discover_proteins(args.benchmark_root)
    print(f"discovered {len(names)} proteins")

    compression = None if args.compression == "none" else args.compression
    mode = "a" if args.output.exists() else "w"
    t_start = time.time()

    with h5py.File(args.output, mode) as f:
        # Root attrs written only on first open.
        if "rmsd_threshold_angstrom" not in f.attrs:
            f.attrs["rmsd_threshold_angstrom"] = args.threshold
            f.attrs["zdock_benchmark"] = "BM4"
            f.attrs["created_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            f.attrs["git_commit"] = _git_commit()

        errors: list[tuple[str, str]] = []
        for i, name in enumerate(names, start=1):
            if name in f and not args.overwrite:
                print(f"[{i}/{len(names)}] {name}: skip (already present)")
                continue
            t0 = time.time()
            try:
                data = _process_protein(
                    name, args.benchmark_root, args.threshold, args.max_poses,
                    sasa_device=sasa_device,
                )
                _write_protein(f, name, data,
                               compression=compression, overwrite=args.overwrite)
                hits = int(data["hit_mask"].sum())
                print(
                    f"[{i}/{len(names)}] {name}: rec={data['rec_xyz'].shape[0]} "
                    f"lig=({data['lig_xyz'].shape[0]}, {data['lig_radius'].shape[0]}) "
                    f"hits={hits} ({time.time()-t0:.1f}s)"
                )
            except Exception as e:
                errors.append((name, str(e)))
                msg = f"[{i}/{len(names)}] {name}: FAILED ({e})"
                if args.skip_errors:
                    print(msg)
                else:
                    raise

        if errors:
            print(f"\n{len(errors)} proteins failed:")
            for n, e in errors:
                print(f"  {n}: {e}")

    print(f"\ndone in {time.time()-t_start:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
