"""Tests for the FFT-based docking search (`zdock.search`).

Phase 1 covers SC-only port; see `PORT_PLAN_FFT.md`. Each test
cross-checks the FFT pipeline against an independent reference:

    - `docking_score_sc_direct` (naive O(V²) cross-correlation)
    - `docking_score_elec` (the Phase-5-tested re-scoring path)

All tests pin float64 on CPU to exercise bit-exact parity.
"""

from __future__ import annotations

import math

import h5py
import numpy as np
import pytest
import torch

from zdock.geom import generate_grid
from zdock.score import docking_score_elec
from zdock.search import (
    _build_ligand_sc_grid_single,
    _build_receptor_sc_grids,
    _rotate_batch,
    _unshift_cyclic,
    docking_score_sc_direct,
    docking_search_sc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sc_only_params():
    """Params that turn `docking_score_elec` into an SC-only evaluator:
    alpha=1, iface=0, beta=0 → total = SC."""
    dtype = torch.float64
    return dict(
        alpha=torch.tensor(1.0, dtype=dtype),
        iface_ij_flat=torch.zeros(144, dtype=dtype),
        beta=torch.tensor(0.0, dtype=dtype),
        charge_score=torch.zeros(11, dtype=dtype),
    )


def _synth_system(seed: int, *, N_rec: int = 8, N_lig: int = 5,
                  rec_spread: float = 2.0, lig_spread: float = 1.5,
                  dtype=torch.float64):
    """Random but reproducible receptor + ligand inputs."""
    g = torch.Generator().manual_seed(seed)
    rec_xyz = torch.randn(N_rec, 3, generator=g, dtype=dtype) * rec_spread
    lig_xyz = torch.randn(N_lig, 3, generator=g, dtype=dtype) * lig_spread
    rec_xyz = rec_xyz - rec_xyz.mean(dim=0)
    lig_xyz = lig_xyz - lig_xyz.mean(dim=0)
    return dict(
        rec_xyz=rec_xyz,
        rec_radius=torch.rand(N_rec, generator=g, dtype=dtype) + 1.0,
        rec_sasa=torch.rand(N_rec, generator=g, dtype=dtype) * 4.0,
        rec_atomtype_id=torch.randint(1, 13, (N_rec,), generator=g, dtype=torch.int64),
        rec_charge_id=torch.randint(1, 12, (N_rec,), generator=g, dtype=torch.int64),
        lig_xyz=lig_xyz,
        lig_radius=torch.rand(N_lig, generator=g, dtype=dtype) + 1.0,
        lig_sasa=torch.rand(N_lig, generator=g, dtype=dtype) * 4.0,
        lig_atomtype_id=torch.randint(1, 13, (N_lig,), generator=g, dtype=torch.int64),
        lig_charge_id=torch.randint(1, 12, (N_lig,), generator=g, dtype=torch.int64),
    )


# ---------------------------------------------------------------------------
# V-direct: FFT vs naive cross-correlation on synthetic grids
# ---------------------------------------------------------------------------


def test_sc_fft_matches_direct_xcorr():
    """The core FFT identity used by docking_search_sc agrees with a
    naive O(V²) direct cross-correlation on small complex grids."""
    torch.manual_seed(0)
    nx, ny, nz = 8, 6, 5
    R_r = torch.randn(nx, ny, nz, dtype=torch.float64)
    R_i = torch.randn(nx, ny, nz, dtype=torch.float64)
    L_r = torch.randn(nx, ny, nz, dtype=torch.float64)
    L_i = torch.randn(nx, ny, nz, dtype=torch.float64)

    # FFT path — same math as the inner loop of docking_search_sc.
    Z_R = R_r + 1j * R_i
    Z_L = L_r + 1j * L_i
    F_Z_R = torch.fft.fftn(Z_R, dim=(-3, -2, -1))
    F_Z_L = torch.fft.fftn(Z_L.conj(), dim=(-3, -2, -1)).conj()
    G = torch.fft.ifftn(F_Z_R * F_Z_L, dim=(-3, -2, -1))
    score_fft = G.real - G.imag

    score_direct = docking_score_sc_direct(R_r, R_i, L_r, L_i)

    assert torch.allclose(score_fft, score_direct, atol=1e-10, rtol=0)


# ---------------------------------------------------------------------------
# V-SC: FFT vs docking_score_elec at identity rotation
# ---------------------------------------------------------------------------


def test_sc_fft_matches_docking_score_elec_synthetic(sc_only_params):
    """At several in-bounds translations, the FFT-search SC score for
    the identity rotation equals `docking_score_elec`'s SC term."""
    dtype = torch.float64
    spacing = 3.0
    sys = _synth_system(seed=3)
    q_id = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=dtype)

    result = docking_search_sc(
        sys["rec_xyz"], sys["rec_radius"], sys["rec_sasa"],
        sys["lig_xyz"], sys["lig_radius"], sys["lig_sasa"],
        quaternions=q_id, spacing=spacing, ntop=400, rot_chunk_size=1,
    )

    for t_cells in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 1), (-2, 2, -1)]:
        t_ang = torch.tensor(
            [t_cells[0] * spacing, t_cells[1] * spacing, t_cells[2] * spacing],
            dtype=dtype,
        )
        lig_pose = (sys["lig_xyz"] + t_ang).unsqueeze(0)
        sc_e = docking_score_elec(
            sys["rec_xyz"], sys["rec_radius"], sys["rec_sasa"],
            sys["rec_atomtype_id"], sys["rec_charge_id"],
            lig_pose, sys["lig_radius"], sys["lig_sasa"],
            sys["lig_atomtype_id"], sys["lig_charge_id"],
            lig_xyz_for_grid=sys["lig_xyz"], spacing=spacing,
            **sc_only_params,
        ).item()
        dists = (result.translations - t_ang).abs().sum(dim=-1)
        idx = dists.argmin().item()
        assert dists[idx].item() < 1e-6, (
            f"translation {t_cells} not in search result"
        )
        sc_fft = result.scores[idx].item()
        assert abs(sc_e - sc_fft) < 1e-8, (
            f"t={t_cells}: elec={sc_e} fft={sc_fft}"
        )


# ---------------------------------------------------------------------------
# V-ROT: non-identity quaternions
# ---------------------------------------------------------------------------


def _axis_angle_q(axis, angle_rad, dtype):
    a = torch.tensor(axis, dtype=dtype)
    a = a / a.norm()
    h = angle_rad / 2
    return torch.tensor(
        [a[0] * math.sin(h), a[1] * math.sin(h), a[2] * math.sin(h), math.cos(h)],
        dtype=dtype,
    )


@pytest.mark.parametrize("axis,angle", [
    ([0.0, 0.0, 0.0], 0.0),
    ([1.0, 0.0, 0.0], math.pi / 2),
    ([0.0, 1.0, 0.0], math.pi / 2),
    ([0.5, 0.3, 0.8], 1.0),
])
def test_sc_fft_matches_elec_non_identity_rotation(axis, angle, sc_only_params):
    """FFT search with non-identity quaternions still matches
    `docking_score_elec` at the rotated-then-translated pose."""
    dtype = torch.float64
    spacing = 3.0
    sys = _synth_system(seed=3)
    if axis == [0.0, 0.0, 0.0]:
        q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=dtype)
    else:
        q = _axis_angle_q(axis, angle, dtype)
    quaternions = q.unsqueeze(0)
    lig_rot = _rotate_batch(sys["lig_xyz"], quaternions)[0]

    result = docking_search_sc(
        sys["rec_xyz"], sys["rec_radius"], sys["rec_sasa"],
        sys["lig_xyz"], sys["lig_radius"], sys["lig_sasa"],
        quaternions=quaternions, spacing=spacing, ntop=400, rot_chunk_size=1,
    )

    for t_cells in [(0, 0, 0), (1, 0, 0), (-1, 1, 0), (2, -2, 1)]:
        t_ang = torch.tensor(
            [t_cells[0] * spacing, t_cells[1] * spacing, t_cells[2] * spacing],
            dtype=dtype,
        )
        lig_pose = (lig_rot + t_ang).unsqueeze(0)
        sc_e = docking_score_elec(
            sys["rec_xyz"], sys["rec_radius"], sys["rec_sasa"],
            sys["rec_atomtype_id"], sys["rec_charge_id"],
            lig_pose, sys["lig_radius"], sys["lig_sasa"],
            sys["lig_atomtype_id"], sys["lig_charge_id"],
            lig_xyz_for_grid=sys["lig_xyz"], spacing=spacing,
            **sc_only_params,
        ).item()
        dists = (result.translations - t_ang).abs().sum(dim=-1)
        idx = dists.argmin().item()
        if dists[idx].item() > 1e-6:
            continue  # not in result
        sc_fft = result.scores[idx].item()
        assert abs(sc_e - sc_fft) < 1e-8, (
            f"axis={axis} angle={angle} t={t_cells}: elec={sc_e} fft={sc_fft}"
        )


# ---------------------------------------------------------------------------
# V-BATCH: rotation chunking bit-exact determinism
# ---------------------------------------------------------------------------


def test_rotation_chunk_size_is_bit_exact():
    """Running the same quaternion list with different rot_chunk_size
    values must produce identical score tensors and identical
    (score, quat_idx, translation) multisets. Detects batching state
    leakage. Uses ntop = grid size to avoid truncation-boundary ties
    (which `torch.topk` resolves non-deterministically)."""
    dtype = torch.float64
    spacing = 3.0
    sys = _synth_system(seed=7, N_rec=10, N_lig=6)

    g = torch.Generator().manual_seed(7)
    q_all = torch.randn(20, 4, generator=g, dtype=dtype)
    q_all = q_all / q_all.norm(dim=-1, keepdim=True)

    _, _, xg, yg, zg = generate_grid(
        sys["rec_xyz"], sys["lig_xyz"], spacing=spacing, dtype=dtype,
    )
    V = xg.numel() * yg.numel() * zg.numel()
    # ntop large enough to cover every (rotation, cell) pair for 20
    # rotations — tie-break at the truncation boundary is then irrelevant.
    ntop = min(V * len(q_all), 500)

    def run(chunk):
        return docking_search_sc(
            sys["rec_xyz"], sys["rec_radius"], sys["rec_sasa"],
            sys["lig_xyz"], sys["lig_radius"], sys["lig_sasa"],
            quaternions=q_all, spacing=spacing, ntop=ntop,
            rot_chunk_size=chunk,
        )

    r1, r5, r13 = run(1), run(5), run(13)

    # Canonical sort by (−score, quat, tx, ty, tz) — deterministic ordering
    # of the same underlying multiset of rows.
    def canon(res):
        keys = torch.stack([
            -res.scores, res.quat_indices.to(res.scores.dtype),
            res.translations[:, 0], res.translations[:, 1],
            res.translations[:, 2],
        ], dim=-1)
        order = sorted(range(len(res.scores)),
                       key=lambda i: tuple(keys[i].tolist()))
        return (res.scores[order], res.quat_indices[order],
                res.translations[order])

    s1, q1, t1 = canon(r1)
    s5, q5, t5 = canon(r5)
    s13, q13, t13 = canon(r13)

    assert (s1 == s5).all() and (s1 == s13).all()
    assert (q1 == q5).all() and (q1 == q13).all()
    assert (t1 == t5).all() and (t1 == t13).all()


# ---------------------------------------------------------------------------
# V-ODD: various grid shapes, in-bounds translations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 3, 7, 11, 17, 23, 42, 101])
def test_sc_fft_matches_elec_across_grid_sizes(seed, sc_only_params):
    """Across 8 seeds producing different even/odd grid shapes, all
    in-bounds small-offset translations match `docking_score_elec` to
    float64 machine precision."""
    dtype = torch.float64
    spacing = 3.0
    sys = _synth_system(seed=seed, N_rec=8, N_lig=5, rec_spread=2.5, lig_spread=1.5)
    q_id = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=dtype)
    _, _, xg, yg, zg = generate_grid(sys["rec_xyz"], sys["lig_xyz"],
                                      spacing=spacing, dtype=dtype)
    nx, ny, nz = xg.numel(), yg.numel(), zg.numel()
    result = docking_search_sc(
        sys["rec_xyz"], sys["rec_radius"], sys["rec_sasa"],
        sys["lig_xyz"], sys["lig_radius"], sys["lig_sasa"],
        quaternions=q_id, spacing=spacing, ntop=nx * ny * nz, rot_chunk_size=1,
    )

    tested = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                t_ang = torch.tensor(
                    [dx * spacing, dy * spacing, dz * spacing], dtype=dtype,
                )
                pos = sys["lig_xyz"] + t_ang
                # skip out-of-bounds translations (A8 wraparound)
                if ((pos[:, 0] < xg[0]).any() or (pos[:, 0] > xg[-1]).any()
                    or (pos[:, 1] < yg[0]).any() or (pos[:, 1] > yg[-1]).any()
                    or (pos[:, 2] < zg[0]).any() or (pos[:, 2] > zg[-1]).any()):
                    continue
                sc_e = docking_score_elec(
                    sys["rec_xyz"], sys["rec_radius"], sys["rec_sasa"],
                    sys["rec_atomtype_id"], sys["rec_charge_id"],
                    pos.unsqueeze(0), sys["lig_radius"], sys["lig_sasa"],
                    sys["lig_atomtype_id"], sys["lig_charge_id"],
                    lig_xyz_for_grid=sys["lig_xyz"], spacing=spacing,
                    **sc_only_params,
                ).item()
                dists = (result.translations - t_ang).abs().sum(dim=-1)
                idx = dists.argmin().item()
                assert dists[idx].item() < 1e-6
                sc_fft = result.scores[idx].item()
                assert abs(sc_e - sc_fft) < 1e-8, (
                    f"seed={seed} t=({dx},{dy},{dz}): elec={sc_e} fft={sc_fft}"
                )
                tested += 1

    assert tested >= 1  # at least one in-bounds translation exists


def test_unshift_cyclic():
    """unshift_cyclic maps DFT bin → signed translation offset
    consistent with torch.fft.ifftn's default (unshifted) output."""
    for N in [4, 5, 7, 8, 9, 16]:
        for k in range(N):
            got = _unshift_cyclic(torch.tensor([k]), N).item()
            expected = k if k <= N // 2 else k - N
            assert got == expected, f"N={N} k={k}: got {got}, expected {expected}"


# ---------------------------------------------------------------------------
# V-1KXQ: real protein (1KXQ phase-5 refs)
# ---------------------------------------------------------------------------


def test_sc_fft_matches_elec_1kxq(refs_root, sc_only_params):
    """Real 3908-atom receptor × 916-atom ligand (1KXQ phase-5 refs):
    FFT SC score at several translations matches `docking_score_elec`'s
    SC term to ~1e-15 relative on a 117×115×129 grid (float64)."""
    dtype = torch.float64
    spacing = 1.2
    path = refs_root / "phase5_scores.h5"
    with h5py.File(path, "r") as f:
        rec_xyz = torch.from_numpy(np.array(f["rec_xyz"], dtype=np.float64)).T.contiguous()
        rec_radius = torch.from_numpy(np.array(f["rec_radius"], dtype=np.float64))
        rec_sasa = torch.from_numpy(np.array(f["rec_sasa"], dtype=np.float64))
        rec_atomtype_id = torch.from_numpy(np.array(f["rec_atomtype_id"], dtype=np.int64))
        rec_charge_id = torch.from_numpy(np.array(f["rec_charge_id"], dtype=np.int64))
        lig_xyz_ref = torch.from_numpy(np.array(f["lig_xyz_for_grid"], dtype=np.float64)).T.contiguous()
        lig_radius = torch.from_numpy(np.array(f["lig_radius"], dtype=np.float64))
        lig_sasa = torch.from_numpy(np.array(f["lig_sasa"], dtype=np.float64))
        lig_atomtype_id = torch.from_numpy(np.array(f["lig_atomtype_id"], dtype=np.int64))
        lig_charge_id = torch.from_numpy(np.array(f["lig_charge_id"], dtype=np.int64))

    _, _, xg, yg, zg = generate_grid(rec_xyz, lig_xyz_ref, spacing=spacing, dtype=dtype)
    nx, ny, nz = xg.numel(), yg.numel(), zg.numel()

    # Build SC grids directly and compute FFT score grid — bypasses top-N.
    R_real, R_imag = _build_receptor_sc_grids(
        rec_xyz, rec_radius, rec_sasa, xg, yg, zg, surface_threshold=1.0,
    )
    lig_surf = lig_sasa > 1.0
    L_real, L_imag = _build_ligand_sc_grid_single(
        lig_xyz_ref, lig_radius, lig_surf, xg, yg, zg,
    )
    Z_R = R_real + 1j * R_imag
    Z_L = L_real + 1j * L_imag
    F_Z_R = torch.fft.fftn(Z_R, dim=(-3, -2, -1))
    F_Z_L = torch.fft.fftn(Z_L.conj(), dim=(-3, -2, -1)).conj()
    G = torch.fft.ifftn(F_Z_R * F_Z_L, dim=(-3, -2, -1))
    score_grid = G.real - G.imag

    for t in [
        (0.0, 0.0, 0.0),
        (spacing, 0.0, 0.0),
        (0.0, spacing, 0.0),
        (0.0, 0.0, spacing),
        (-spacing, spacing, -spacing),
        (2 * spacing, -spacing, 0.0),
    ]:
        t_ang = torch.tensor(t, dtype=dtype)
        lig_pose = (lig_xyz_ref + t_ang).unsqueeze(0)
        sc_e = docking_score_elec(
            rec_xyz, rec_radius, rec_sasa, rec_atomtype_id, rec_charge_id,
            lig_pose, lig_radius, lig_sasa, lig_atomtype_id, lig_charge_id,
            lig_xyz_for_grid=lig_xyz_ref, spacing=spacing,
            **sc_only_params,
        ).item()
        # bin index for translation t
        tx = int(round(t[0] / spacing)) % nx
        ty = int(round(t[1] / spacing)) % ny
        tz = int(round(t[2] / spacing)) % nz
        sc_fft = score_grid[tx, ty, tz].item()
        rel = abs(sc_e - sc_fft) / (abs(sc_e) + 1)
        assert rel < 1e-10, f"t={t}: elec={sc_e} fft={sc_fft} rel={rel}"
