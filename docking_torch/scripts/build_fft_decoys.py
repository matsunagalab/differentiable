"""Build a FFT-generated decoy dataset with DockQ labels.

For each protein in `bm4_full.h5`, run `docking_search` with a
configurable rotation grid, reconstruct the top-N poses in the
receptor's raw frame, and compute DockQ v2 (atom-level approximation)
against a pseudo-native reference. Write everything to an output h5
suitable for training via `examples/06_train_dockq_fft.py`.

Pseudo-native = BM4 decoy with smallest stored RMSD per protein.
This avoids rebuilding bm4_full.h5 with crystal coordinates while
giving a near-native (≤ 2.5 Å for most complexes) anchor for DockQ.

Output h5 layout (one group per protein):

    /<PROTEIN>/
        lig_xyz         (F, N_lig, 3) float32  pose coords in raw frame
        score           (F,)          float32  FFT scorer value
        rotation_quat   (F, 4)        float32
        translation     (F, 3)        float32  Å in decentered frame
        fnat            (F,)          float32
        i_rmsd          (F,)          float32  Å
        l_rmsd          (F,)          float32  Å
        dockq           (F,)          float32  [0, 1]
        attrs:
            pseudo_native_bm4_idx
            pseudo_native_rmsd_to_crystal

Root attrs: rotation_desc, ntop, spacing, alpha, beta, params_ckpt.

Examples:

    # Quick subset, single GPU
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/build_fft_decoys.py \
        --proteins 1PPE 2SIC 1R0R --n-rotations 2048 --ntop 1000 \
        --out out/fft_decoys_smoke.h5

    # Full BM4, multi-GPU pool
    uv run python scripts/build_fft_decoys.py \
        --gpus 0,1,2,3 --n-rotations 4096 --ntop 2000

    # ZDOCK-comparable 6° Euler grid with trained scorer
    uv run python scripts/build_fft_decoys.py --gpus 0,1,2,3 \
        --euler-deg 6.0 --ntop 500 \
        --params-ckpt out/trained_params_rank.pt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from zdock.atomtypes import charge_score as default_charge_score_lut
from zdock.atomtypes import iface_ij as default_iface_ij
from zdock.data import list_proteins
from zdock.dockq import dockq_batch
from zdock.rotation_grid import euler_quaternions, random_quaternions
from zdock.search import (
    _rotate_batch,
    docking_search,
    prepare_ligand,
)

DEFAULT_H5 = Path(__file__).resolve().parent.parent / "datasets" / "bm4_full.h5"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "out" / "fft_decoys_dockq.h5"


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
    if args.euler_deg is not None:
        return euler_quaternions(deg=args.euler_deg, device=device, dtype=dtype)
    return random_quaternions(
        args.n_rotations, seed=args.seed, device=device, dtype=dtype,
    )


def load_params(
    ckpt_path: Path | None, *,
    device: torch.device, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if ckpt_path is None:
        return (
            torch.tensor(0.01, dtype=dtype, device=device),
            default_iface_ij(device=device, dtype=dtype, flat=True).clone(),
            torch.tensor(3.0, dtype=dtype, device=device),
            default_charge_score_lut(device=device, dtype=dtype).clone(),
        )
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    return (
        ckpt["alpha"].to(device=device, dtype=dtype),
        ckpt["iface"].to(device=device, dtype=dtype),
        torch.tensor(3.0, dtype=dtype, device=device),
        ckpt["charge"].to(device=device, dtype=dtype),
    )


def run_one_protein(
    protein_id: str, h5_in: h5py.File,
    *,
    quaternions: torch.Tensor,
    alpha: torch.Tensor, iface_flat: torch.Tensor,
    beta: torch.Tensor, charge_lut: torch.Tensor,
    spacing: float, ntop: int, rot_chunk_size: int,
    device: torch.device, dtype: torch.dtype,
) -> dict | None:
    g = h5_in[protein_id]
    rec_xyz_raw = torch.as_tensor(g["rec_xyz"][()], dtype=dtype, device=device)
    rec_radius = torch.as_tensor(g["rec_radius"][()], dtype=dtype, device=device)
    rec_sasa = torch.as_tensor(g["rec_sasa"][()], dtype=dtype, device=device)
    rec_aid = torch.as_tensor(g["rec_atomtype_id"][()], dtype=torch.int64, device=device)
    rec_cid = torch.as_tensor(g["rec_charge_id"][()], dtype=torch.int64, device=device)
    lig_all = torch.as_tensor(g["lig_xyz"][()], dtype=dtype, device=device)
    lig_radius = torch.as_tensor(g["lig_radius"][()], dtype=dtype, device=device)
    lig_sasa = torch.as_tensor(g["lig_sasa"][()], dtype=dtype, device=device)
    lig_aid = torch.as_tensor(g["lig_atomtype_id"][()], dtype=torch.int64, device=device)
    lig_cid = torch.as_tensor(g["lig_charge_id"][()], dtype=torch.int64, device=device)
    rmsd_bm4 = torch.as_tensor(g["rmsd"][()], dtype=dtype, device=device)

    # Pseudo-native = BM4 decoy with smallest stored RMSD.
    nn_idx = int(rmsd_bm4.argmin().item())
    pseudo_native_raw = lig_all[nn_idx]  # (N_lig, 3) in receptor raw frame
    pseudo_native_rmsd = float(rmsd_bm4[nn_idx].item())

    rec_com = rec_xyz_raw.mean(dim=0)
    rec_xyz = rec_xyz_raw - rec_com
    lig_xyz_ref = prepare_ligand(lig_all[0], lig_aid)

    # Run FFT search with OOM retry cascade.
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

    # Reconstruct ligand poses in the raw (BM4) frame for DockQ comparison.
    q_per_pose = quaternions[result.quat_indices]                 # (F, 4)
    lig_rot_all = _rotate_batch(lig_xyz_ref, q_per_pose)          # (F, N_lig, 3)
    lig_xyz_decoy_raw = lig_rot_all + result.translations.unsqueeze(-2) + rec_com

    # DockQ uses the raw-frame coords directly (shares frame with rec_xyz_raw
    # and pseudo_native_raw — the plan's "shared receptor frame" assumption
    # that lets us drop Kabsch alignment).
    with torch.no_grad():
        dq = dockq_batch(
            rec_xyz_raw, lig_xyz_decoy_raw, pseudo_native_raw,
        )

    return {
        "lig_xyz": lig_xyz_decoy_raw.cpu().to(torch.float32).numpy(),
        "score": result.scores.cpu().to(torch.float32).numpy(),
        "rotation_quat": q_per_pose.cpu().to(torch.float32).numpy(),
        "translation": result.translations.cpu().to(torch.float32).numpy(),
        "fnat": dq.fnat.cpu().to(torch.float32).numpy(),
        "i_rmsd": dq.i_rmsd.cpu().to(torch.float32).numpy(),
        "l_rmsd": dq.l_rmsd.cpu().to(torch.float32).numpy(),
        "dockq": dq.dockq.cpu().to(torch.float32).numpy(),
        "_pseudo_native_bm4_idx": nn_idx,
        "_pseudo_native_rmsd_to_crystal": pseudo_native_rmsd,
        "_final_chunk": chunk,
    }


def _merge_h5(
    tmp_files: list[Path], out_path: Path, root_attrs: dict,
) -> int:
    written = 0
    with h5py.File(out_path, "a") as h5_out:
        for k, v in root_attrs.items():
            h5_out.attrs[k] = v
        for tmp in tmp_files:
            if not tmp.exists():
                continue
            with h5py.File(tmp, "r") as h5_in:
                for prot in h5_in:
                    if prot in h5_out:
                        del h5_out[prot]
                    h5_in.copy(prot, h5_out)
                    written += 1
    return written


def dispatch_multi_gpu(args: argparse.Namespace) -> None:
    """Dynamic 1-complex-per-GPU dispatcher (same pattern as
    examples/05_fft_generate_decoys.py but for the DockQ build).
    """
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--gpus must list at least one GPU index")
    all_names = list_proteins(args.h5)
    names = args.proteins if args.proteins else all_names
    missing = [p for p in names if p not in all_names]
    if missing:
        raise SystemExit(f"proteins not in {args.h5}: {missing}")
    print(f"dispatching {len(names)} proteins across {len(gpus)} GPUs "
          f"(1 complex / 1 GPU, dynamic assignment)")

    drop_flags_1 = {"--gpus", "--device", "--out"}
    drop_flags_multi = {"--proteins"}
    base_argv: list[str] = []
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a in drop_flags_1:
            i += 2
            continue
        if a in drop_flags_multi:
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith("--"):
                i += 1
            continue
        base_argv.append(a)
        i += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pending: list[str] = list(names)
    active: dict[str, tuple | None] = {g: None for g in gpus}
    tmp_files: list[Path] = []
    done, failed = 0, []
    total_start = time.time()

    def launch(gpu: str, protein: str) -> tuple:
        tmp = args.out.with_suffix(f".{protein}.h5.tmp")
        if tmp.exists():
            tmp.unlink()
        child = [sys.executable, sys.argv[0]] + base_argv + [
            "--device", "cuda",
            "--out", str(tmp),
            "--proteins", protein,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        t0 = time.time()
        p = subprocess.Popen(
            child, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
        tmp_files.append(tmp)
        print(f"[GPU {gpu}] → {protein} (pending={len(pending)})")
        return (protein, p, tmp, t0)

    for gpu in gpus:
        if not pending:
            break
        active[gpu] = launch(gpu, pending.pop(0))

    while any(v is not None for v in active.values()):
        made_progress = False
        for gpu, slot in active.items():
            if slot is None:
                continue
            protein, proc, tmp, t0 = slot
            if proc.poll() is None:
                continue
            dt = time.time() - t0
            rc = proc.returncode
            if rc == 0:
                print(f"[GPU {gpu}] ✓ {protein} ({dt:.1f}s)")
                done += 1
            else:
                failed.append(protein)
                print(f"[GPU {gpu}] ✗ {protein} exit={rc} ({dt:.1f}s)")
            active[gpu] = None
            made_progress = True
            if pending:
                active[gpu] = launch(gpu, pending.pop(0))
        if not made_progress:
            time.sleep(0.5)

    total = time.time() - total_start
    print(f"\nAll workers done: {done} OK, {len(failed)} failed "
          f"({total:.1f}s = {total/60:.1f} min)")
    if failed:
        print(f"  failed: {failed}")

    rot_desc = (
        f"random n={args.n_rotations} (seed={args.seed})"
        if args.euler_deg is None else f"euler deg={args.euler_deg}"
    )
    root_attrs = dict(
        rotation_desc=rot_desc,
        ntop=int(args.ntop),
        spacing=float(args.spacing),
    )
    if args.params_ckpt is not None:
        root_attrs["params_ckpt"] = str(args.params_ckpt)
    if args.out.exists():
        args.out.unlink()
    n_written = _merge_h5(tmp_files, args.out, root_attrs)
    for tmp in tmp_files:
        if tmp.exists():
            tmp.unlink()
    print(f"merged {n_written} protein groups → {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=DEFAULT_H5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--proteins", nargs="+", default=None)
    rot_g = ap.add_mutually_exclusive_group()
    rot_g.add_argument("--n-rotations", type=int, default=4096)
    rot_g.add_argument("--euler-deg", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ntop", type=int, default=2000)
    ap.add_argument("--spacing", type=float, default=1.2)
    ap.add_argument("--rot-chunk-size", type=int, default=64)
    ap.add_argument("--params-ckpt", type=Path, default=None)
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU indices, e.g. '0,1,2,3'. "
                         "If set, dispatch 1 complex per GPU dynamically.")
    args = ap.parse_args()

    if not args.h5.exists():
        raise SystemExit(f"input h5 not found: {args.h5}")

    if args.gpus:
        dispatch_multi_gpu(args)
        return

    device = resolve_device(args.device)
    dtype = default_dtype(device)
    print(f"device={device} dtype={dtype}")

    quaternions = build_quaternions(args, device=device, dtype=dtype)
    rot_desc = (
        f"random n={quaternions.shape[0]} (seed={args.seed})"
        if args.euler_deg is None
        else f"euler deg={args.euler_deg} n={quaternions.shape[0]}"
    )
    print(f"rotation grid: {rot_desc}")

    alpha, iface_flat, beta, charge_lut = load_params(
        args.params_ckpt, device=device, dtype=dtype,
    )
    print(f"scorer: alpha={alpha.item():.4e} beta={beta.item():.4e} "
          f"{'(trained)' if args.params_ckpt else '(Julia defaults)'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_names = list_proteins(args.h5)
    names = args.proteins if args.proteins else all_names
    missing = [p for p in names if p not in all_names]
    if missing:
        raise SystemExit(f"proteins not in {args.h5}: {missing}")
    print(f"proteins to process: {len(names)}")

    total_start = time.time()
    with h5py.File(args.h5, "r") as h5_in, h5py.File(args.out, "a") as h5_out:
        h5_out.attrs["rotation_desc"] = rot_desc
        h5_out.attrs["n_rotations"] = int(quaternions.shape[0])
        h5_out.attrs["ntop"] = int(args.ntop)
        h5_out.attrs["spacing"] = float(args.spacing)
        h5_out.attrs["alpha"] = float(alpha.item())
        h5_out.attrs["beta"] = float(beta.item())
        if args.params_ckpt is not None:
            h5_out.attrs["params_ckpt"] = str(args.params_ckpt)

        for i, prot in enumerate(names, 1):
            if prot in h5_out and "dockq" in h5_out[prot]:
                print(f"[{i:>3}/{len(names)}] {prot}: already done, skip")
                continue
            t0 = time.time()
            result = run_one_protein(
                prot, h5_in,
                quaternions=quaternions,
                alpha=alpha, iface_flat=iface_flat,
                beta=beta, charge_lut=charge_lut,
                spacing=args.spacing, ntop=args.ntop,
                rot_chunk_size=args.rot_chunk_size,
                device=device, dtype=dtype,
            )
            dt = time.time() - t0
            if result is None:
                print(f"[{i:>3}/{len(names)}] {prot}: OOM after retry, skip")
                continue

            chunk = result.pop("_final_chunk")
            best_dq = float(np.max(result["dockq"]))
            top_dq = float(result["dockq"][0])
            n_positive = int(np.sum(result["dockq"] >= 0.23))
            print(
                f"[{i:>3}/{len(names)}] {prot}: {dt:>5.1f}s chunk={chunk} "
                f"top-score DockQ={top_dq:.2f} best-DockQ={best_dq:.2f} "
                f"n_positive={n_positive}/{len(result['dockq'])}"
            )

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
    print(f"\nDone in {total:.1f}s ({total/60:.1f} min) → {args.out}")


if __name__ == "__main__":
    main()
