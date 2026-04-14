"""CPU vs GPU benchmark for docking_score_elec (forward + backward).

Uses the phase5 1KXQ reference inputs and replicates the 10 ligand poses
along the frame axis to sweep F ∈ {10, 40, 100}.

Run (from docking_torch/):
    CUDA_VISIBLE_DEVICES=2 uv run python benchmark_gpu.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import h5py
import numpy as np
import torch

from zdock.score import docking_score_elec

REFS = Path(__file__).resolve().parent.parent / "docking" / "tests" / "refs" / "1KXQ"


def _load_h5(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = f[k][()]
    return out


def _2d(a):
    a = np.asarray(a)
    return a.T if (a.ndim == 2 and a.shape[0] == 3) else a


def _3d(a):
    a = np.asarray(a)
    return a.transpose(2, 1, 0) if (a.ndim == 3 and a.shape[0] == 3) else a


def _prepare(device, dtype, F_target):
    ref = _load_h5(REFS / "phase5_scores.h5")
    t = lambda x, dt=dtype: torch.as_tensor(np.asarray(x), device=device, dtype=dt)

    alpha = t(float(ref["alpha"])).clone().requires_grad_(True)
    beta = t(float(ref["beta"])).clone().requires_grad_(True)
    iface = t(ref["iface_ij_flat"]).clone().requires_grad_(True)
    charge_score = t(ref["charge_score"]).clone().requires_grad_(True)

    rec_xyz = t(_2d(ref["rec_xyz"]))
    rec_radius = t(ref["rec_radius"])
    rec_sasa = t(ref["rec_sasa"])
    rec_atomtype = t(ref["rec_atomtype_id"], torch.int64)
    rec_charge = t(ref["rec_charge_id"], torch.int64)

    lig_xyz = t(_3d(ref["lig_xyz"]))  # (10, N_lig, 3)
    # Replicate to reach F_target (tile along frame dim).
    reps = (F_target + lig_xyz.shape[0] - 1) // lig_xyz.shape[0]
    lig_xyz = lig_xyz.repeat(reps, 1, 1)[:F_target].contiguous()
    lig_radius = t(ref["lig_radius"])
    lig_sasa = t(ref["lig_sasa"])
    lig_atomtype = t(ref["lig_atomtype_id"], torch.int64)
    lig_charge = t(ref["lig_charge_id"], torch.int64)

    return dict(
        rec_xyz=rec_xyz, rec_radius=rec_radius, rec_sasa=rec_sasa,
        rec_atomtype_id=rec_atomtype, rec_charge_id=rec_charge,
        lig_xyz=lig_xyz, lig_radius=lig_radius, lig_sasa=lig_sasa,
        lig_atomtype_id=lig_atomtype, lig_charge_id=lig_charge,
        alpha=alpha, iface_ij_flat=iface, beta=beta, charge_score=charge_score,
    )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(device_name, F, n_warmup=2, n_iters=5):
    device = torch.device(device_name)
    dtype = torch.float64 if device.type == "cpu" else torch.float32
    inp = _prepare(device, dtype, F)

    params = dict(alpha=inp["alpha"], iface_ij_flat=inp["iface_ij_flat"],
                  beta=inp["beta"], charge_score=inp["charge_score"])
    tensor_inp = {k: v for k, v in inp.items() if k not in params}

    def _call():
        return docking_score_elec(**tensor_inp, **params)

    # --- forward ---
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = _call()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = _call()
    _sync(device)
    fwd_ms = (time.perf_counter() - t0) * 1000 / n_iters

    # --- forward + backward ---
    for _ in range(n_warmup):
        for p in params.values():
            if p.grad is not None:
                p.grad = None
        s = _call().sum()
        s.backward()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        for p in params.values():
            if p.grad is not None:
                p.grad = None
        s = _call().sum()
        s.backward()
    _sync(device)
    fb_ms = (time.perf_counter() - t0) * 1000 / n_iters

    return fwd_ms, fb_ms


def main():
    print(f"torch {torch.__version__}, CUDA avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}  "
              f"(visible index 0 -> physical {os.environ.get('CUDA_VISIBLE_DEVICES', '?')})")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    Fs = [10, 40, 100]
    print(f"\n{'device':<8}{'dtype':<10}{'F':>5}  {'fwd (ms)':>10}  {'fwd+bwd (ms)':>14}")
    print("-" * 55)
    results = {}
    for dev in devices:
        for F in Fs:
            fwd, fb = bench(dev, F)
            dt_str = "float64" if dev == "cpu" else "float32"
            print(f"{dev:<8}{dt_str:<10}{F:>5}  {fwd:>10.1f}  {fb:>14.1f}")
            results[(dev, F)] = (fwd, fb)

    if "cuda" in devices:
        print("\nspeedup (CPU / CUDA):")
        print(f"{'F':>5}  {'fwd':>8}  {'fwd+bwd':>10}")
        for F in Fs:
            c_fwd, c_fb = results[("cpu", F)]
            g_fwd, g_fb = results[("cuda", F)]
            print(f"{F:>5}  {c_fwd/g_fwd:>7.2f}x  {c_fb/g_fb:>9.2f}x")


if __name__ == "__main__":
    main()
