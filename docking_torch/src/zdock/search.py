"""FFT-based docking search — pose generation side of the PyTorch ZDOCK
pipeline.

For each quaternion in a caller-supplied rotation grid, we scatter the
rotated ligand onto a complex SC grid, FFT, multiply with the receptor's
precomputed FFT, inverse-FFT, and extract top-N translation peaks. All
translations for a fixed rotation are evaluated in a single FFT pair.

Convention (audited — see `PORT_PLAN_FFT.md`):

    For real grids:
        ifft(fft(R) · conj(fft(L)))[t] = Σ_m R[m] · L[m − t]
    i.e. peak at t = t0 means "the ligand is best placed at translation
    +t0 from its current position." The decoder ADDs t0 to the ligand
    coordinates to produce the docked pose.

    For the complex SC grid Z_R = R_real + i·R_imag, Z_L = L_real +
    i·L_imag, the docking_score_elec's SC term at a fixed pose is
        Σ_cell [real(Z_R · Z_L) − imag(Z_R · Z_L)]
    The FFT generalisation over translations uses the complex
    cross-correlation G = ifft(fft(Z_R) · conj(fft(conj(Z_L)))), which
    for any complex Z_L equals Σ_m Z_R[m] · Z_L[m − t] (no conjugation
    of Z_L inside the sum — the outer conj+conj cancels for this
    purpose). Then score_sc(t) = real(G[t]) − imag(G[t]).

This file implements Phase 1 of `PORT_PLAN_FFT.md`: SC term only. DS,
IFACE, ELEC follow in Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import math

from .geom import generate_grid
from .score import (
    _assign_sc_plus,
    _assign_sc_minus,
    _grouped_spread_neighbors_add,
)


@dataclass
class DockingResultSC:
    """Phase-1 result: SC-only scores and decoded poses.

    Attributes:
        scores: (ntop,) float, sorted descending (highest SC first).
        quat_indices: (ntop,) int64, index into the caller's quaternion
            grid that produced each top pose.
        translations: (ntop, 3) float, cartesian ligand translation in
            Å to apply on top of the rotation. Decoding: the docked
            ligand coordinates are ``rotate(ref_lig_xyz, q) +
            translations[i]``.
    """
    scores: torch.Tensor
    quat_indices: torch.Tensor
    translations: torch.Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotate_batch(lig_xyz: torch.Tensor, quaternions: torch.Tensor) -> torch.Tensor:
    """Rotate an (N, 3) set of ligand coords by each of B quaternions.

    Uses the same rotation-matrix entries as `geom.rotate` (which is in
    turn verbatim from `docking.jl` lines 1469–1490) so poses agree with
    the existing `docking_score_elec` pipeline.

    Returns (B, N, 3).
    """
    q = quaternions
    q1, q2, q3, q4 = q.unbind(-1)

    r1 = 1.0 - 2.0 * q2 * q2 - 2.0 * q3 * q3
    r2 = 2.0 * (q1 * q2 + q3 * q4)
    r3 = 2.0 * (q1 * q3 - q2 * q4)
    r4 = 2.0 * (q1 * q2 - q3 * q4)
    r5 = 1.0 - 2.0 * q1 * q1 - 2.0 * q3 * q3
    r6 = 2.0 * (q2 * q3 + q1 * q4)
    r7 = 2.0 * (q1 * q3 + q2 * q4)
    r8 = 2.0 * (q2 * q3 - q1 * q4)
    r9 = 1.0 - 2.0 * q1 * q1 - 2.0 * q2 * q2

    # (B, 3, 3)
    R_mat = torch.stack(
        [torch.stack([r1, r2, r3], dim=-1),
         torch.stack([r4, r5, r6], dim=-1),
         torch.stack([r7, r8, r9], dim=-1)],
        dim=-2,
    )
    # lig_xyz: (N, 3) → broadcast
    # (B, 3, 3) @ (3, N) → (B, 3, N) → transpose → (B, N, 3)
    return torch.einsum("bij,nj->bni", R_mat, lig_xyz)


def _build_receptor_sc_grids(
    rec_xyz: torch.Tensor,
    rec_radius: torch.Tensor,
    rec_sasa: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    surface_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter receptor atoms into the SC real/imag grids."""
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = rec_xyz.device
    dtype = rec_xyz.dtype
    R_real = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    R_imag = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    rec_surf = rec_sasa > surface_threshold
    _assign_sc_plus(R_real, rec_xyz, rec_radius, rec_surf,
                    x_grid, y_grid, z_grid, receptor=True)
    _assign_sc_minus(R_imag, rec_xyz, rec_radius, rec_surf,
                     x_grid, y_grid, z_grid, receptor=True)
    return R_real, R_imag


def _build_ligand_sc_grid_single(
    lig_xyz: torch.Tensor,         # (N, 3)
    lig_radius: torch.Tensor,      # (N,)
    lig_surf: torch.Tensor,        # (N,) bool
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replicate `docking_score_elec`'s ligand SC construction exactly
    (score.py lines 281–313). Produces the real and imag SC grids for
    one ligand pose. Key differences from `_assign_sc_minus(receptor=False)`:

        * Surface atoms contribute 1.0 to the real grid and 3.5 to the
          imag grid (not 12.25).
        * Core atoms contribute 1.0 to real and 12.25 to imag.
        * Core overwrites surface in imag (core takes precedence where
          both cover a cell).
        * Layer boundaries come from (cnt > 0) indicator unions, not
          substitute writes.

    Matching this exactly is required for Phase 1 V-SC parity against
    `docking_score_elec`.
    """
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz.device
    dtype = lig_xyz.dtype
    core = ~lig_surf

    def sc_union(xyz_sub, rcut_sub):
        """Binary indicator of cells within rcut_sub of any atom in xyz_sub."""
        cnt = torch.zeros((1, nx, ny, nz), device=device, dtype=dtype)
        if xyz_sub.shape[0] == 0:
            return cnt[0]
        group = torch.zeros(xyz_sub.shape[0], device=device, dtype=torch.long)
        weights = torch.ones(xyz_sub.shape[0], device=device, dtype=dtype)
        _grouped_spread_neighbors_add(
            cnt, xyz_sub, group, weights, rcut_sub, x_grid, y_grid, z_grid,
        )
        return (cnt[0] > 0).to(dtype)

    L_real = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
    L_imag = torch.zeros((nx, ny, nz), device=device, dtype=dtype)

    surf_idx = lig_surf.nonzero(as_tuple=True)[0]
    core_idx = core.nonzero(as_tuple=True)[0]

    if surf_idx.numel() > 0:
        lay1 = sc_union(lig_xyz[surf_idx], lig_radius[surf_idx])
        L_real = torch.maximum(L_real, lay1)
    if core_idx.numel() > 0:
        lay2 = sc_union(
            lig_xyz[core_idx], lig_radius[core_idx] * math.sqrt(1.5),
        )
        L_real = torch.maximum(L_real, lay2)

    if surf_idx.numel() > 0:
        lay1 = sc_union(lig_xyz[surf_idx], lig_radius[surf_idx])
        L_imag = torch.where(lay1 > 0, lay1 * 3.5, L_imag)
    if core_idx.numel() > 0:
        lay2 = sc_union(
            lig_xyz[core_idx], lig_radius[core_idx] * math.sqrt(1.5),
        )
        L_imag = torch.where(lay2 > 0, lay2 * 12.25, L_imag)

    return L_real, L_imag


def _build_ligand_sc_grids_batch(
    lig_xyz_rot: torch.Tensor,   # (B, N, 3)
    lig_radius: torch.Tensor,
    lig_sasa: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
    *,
    surface_threshold: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter each rotated ligand in the batch into its own SC grid.

    Phase-1 implementation: Python-loop over B, replicating
    `docking_score_elec`'s internal ligand SC construction. Batching
    the scatter is a Phase-3 optimisation.

    Returns (L_real, L_imag), each (B, nx, ny, nz).
    """
    B = lig_xyz_rot.shape[0]
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()
    device = lig_xyz_rot.device
    dtype = lig_xyz_rot.dtype
    L_real = torch.zeros((B, nx, ny, nz), device=device, dtype=dtype)
    L_imag = torch.zeros((B, nx, ny, nz), device=device, dtype=dtype)
    lig_surf = lig_sasa > surface_threshold
    for b in range(B):
        Lr, Li = _build_ligand_sc_grid_single(
            lig_xyz_rot[b], lig_radius, lig_surf, x_grid, y_grid, z_grid,
        )
        L_real[b] = Lr
        L_imag[b] = Li
    return L_real, L_imag


def _unshift_cyclic(idx: torch.Tensor, N: int) -> torch.Tensor:
    """Convert 0-indexed DFT bin to signed translation offset.

    DFT output at bin k represents translation:
        +k cells    if k <= N/2
        k − N cells if k >  N/2
    """
    half = N // 2
    return torch.where(idx <= half, idx, idx - N)


# ---------------------------------------------------------------------------
# Top-level search API
# ---------------------------------------------------------------------------


def docking_search_sc(
    rec_xyz: torch.Tensor,
    rec_radius: torch.Tensor,
    rec_sasa: torch.Tensor,
    lig_xyz_ref: torch.Tensor,      # (N_lig, 3) — reference ligand, decentered
    lig_radius: torch.Tensor,
    lig_sasa: torch.Tensor,
    quaternions: torch.Tensor,      # (R, 4)
    *,
    spacing: float = 3.0,
    surface_threshold: float = 1.0,
    ntop: int = 100,
    rot_chunk_size: int = 16,
) -> DockingResultSC:
    """FFT-based SC-only docking search.

    For each quaternion in `quaternions`, rotate `lig_xyz_ref` and
    evaluate the SC term at every translation in the receptor grid in
    one FFT per rotation. Keep the top `ntop` (quat, translation) pairs
    across all rotations.

    Inputs:
        rec_xyz: (N_rec, 3) decentered receptor atoms.
        rec_radius, rec_sasa: per-atom receptor features.
        lig_xyz_ref: (N_lig, 3) decentered, orient-aligned ligand atoms.
            The FFT search does not re-apply orient(); pass coords in
            the same frame `docking_score_elec` expects.
        lig_radius, lig_sasa: per-atom ligand features.
        quaternions: (R, 4) caller-supplied rotation grid. Same
            convention as `geom.rotate`.

    Kwargs:
        spacing: grid spacing (Å). Defaults to 3.0 matching
            `docking_score_elec`.
        surface_threshold: SASA cutoff for surface vs core. Default 1.0
            matches `docking_score_elec`.
        ntop: number of top poses to return across all rotations.
        rot_chunk_size: rotations processed per FFT batch (VRAM knob).

    Returns `DockingResultSC`.
    """
    device = rec_xyz.device
    dtype = rec_xyz.dtype
    n_rot = quaternions.shape[0]

    # 1. Build grid axes exactly as `docking_score_elec` does.
    #    generate_grid uses the ligand's bounding box for padding, so we
    #    pass the reference (decentered) ligand — every rotation lives in
    #    this same box because rotation preserves extent on average
    #    (slight variance handled by the padding).
    _, _, x_grid, y_grid, z_grid = generate_grid(
        rec_xyz, lig_xyz_ref, spacing=spacing, device=device, dtype=dtype,
    )
    nx, ny, nz = x_grid.numel(), y_grid.numel(), z_grid.numel()

    # 2. Receptor-side FFT (done once).
    R_real, R_imag = _build_receptor_sc_grids(
        rec_xyz, rec_radius, rec_sasa, x_grid, y_grid, z_grid,
        surface_threshold=surface_threshold,
    )
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    Z_R = (R_real + 1j * R_imag).to(complex_dtype)
    F_Z_R = torch.fft.fftn(Z_R, dim=(-3, -2, -1))

    # 3. Running top-ntop buffer across rotations.
    buf_scores = torch.full((ntop,), float("-inf"), device=device, dtype=dtype)
    buf_quat = torch.zeros((ntop,), device=device, dtype=torch.long)
    buf_flat = torch.zeros((ntop,), device=device, dtype=torch.long)

    # 4. Per-rotation FFT loop (chunked for VRAM).
    for chunk_start in range(0, n_rot, rot_chunk_size):
        chunk_end = min(chunk_start + rot_chunk_size, n_rot)
        B = chunk_end - chunk_start

        # Rotate ligand per quaternion in the chunk.
        lig_rot = _rotate_batch(lig_xyz_ref, quaternions[chunk_start:chunk_end])

        # Scatter each rotated ligand into SC grids.
        L_real, L_imag = _build_ligand_sc_grids_batch(
            lig_rot, lig_radius, lig_sasa, x_grid, y_grid, z_grid,
            surface_threshold=surface_threshold,
        )
        Z_L = (L_real + 1j * L_imag).to(complex_dtype)

        # Per-rotation FFT. The identity used here:
        #   G[t] = Σ_m Z_R[m] · Z_L[m − t]
        #        = ifft(fft(Z_R) · conj(fft(conj(Z_L))))[t]
        # which for real Z_R = R_r + i R_i and Z_L = L_r + i L_i gives
        # score_sc(t) = Re(G[t]) − Im(G[t]), matching
        # `docking_score_elec`'s pointwise SC combination at t=0.
        F_Z_L = torch.fft.fftn(Z_L.conj(), dim=(-3, -2, -1)).conj()
        G = torch.fft.ifftn(F_Z_R.unsqueeze(0) * F_Z_L, dim=(-3, -2, -1))
        score_grid = G.real - G.imag           # (B, nx, ny, nz), real dtype

        # 5. Per-rotation top-k, merge with running buffer.
        score_flat = score_grid.reshape(B, -1)  # (B, V)
        k_per_rot = min(ntop, score_flat.shape[1])
        top_vals, top_idx = torch.topk(score_flat, k=k_per_rot, dim=1)

        # Flatten chunk's (B × k_per_rot) candidates.
        cand_scores = top_vals.reshape(-1)
        cand_quat = (
            chunk_start
            + torch.arange(B, device=device).unsqueeze(-1).expand(-1, k_per_rot)
        ).reshape(-1).to(torch.long)
        cand_flat = top_idx.reshape(-1).to(torch.long)

        # Merge and re-prune.
        all_scores = torch.cat([buf_scores, cand_scores])
        all_quat = torch.cat([buf_quat, cand_quat])
        all_flat = torch.cat([buf_flat, cand_flat])
        new_vals, new_idx = torch.topk(all_scores, k=ntop)
        buf_scores = new_vals
        buf_quat = all_quat[new_idx]
        buf_flat = all_flat[new_idx]

    # 6. Decode flat cell indices to signed cartesian translations.
    V = nx * ny * nz
    tz = buf_flat % nz
    ty = (buf_flat // nz) % ny
    tx = (buf_flat // (ny * nz)) % nx
    tx_s = _unshift_cyclic(tx, nx).to(dtype) * spacing
    ty_s = _unshift_cyclic(ty, ny).to(dtype) * spacing
    tz_s = _unshift_cyclic(tz, nz).to(dtype) * spacing
    translations = torch.stack([tx_s, ty_s, tz_s], dim=-1)

    return DockingResultSC(
        scores=buf_scores,
        quat_indices=buf_quat,
        translations=translations,
    )


# ---------------------------------------------------------------------------
# Direct (non-FFT) reference score, for tests. O(N³) per translation, so
# only usable on tiny grids.
# ---------------------------------------------------------------------------


def docking_score_sc_direct(
    R_real: torch.Tensor, R_imag: torch.Tensor,
    L_real: torch.Tensor, L_imag: torch.Tensor,
) -> torch.Tensor:
    """Naive O(V²) cross-correlation for test reference.

    score_sc(t) = Σ_cell [R_r·L_r − R_i·L_i − R_r·L_i − R_i·L_r]
                         evaluated with L shifted by +t.

    Returns a (nx, ny, nz) tensor of scores indexed cyclically (index 0 =
    zero translation, large indices wrap around to negative translations,
    matching `torch.fft.ifftn` output ordering).
    """
    nx, ny, nz = R_real.shape
    out = torch.zeros_like(R_real)
    # Loop over all translations. O(V²) — only for tiny grids.
    for tx in range(nx):
        for ty in range(ny):
            for tz in range(nz):
                # L shifted cyclically by (tx, ty, tz): L_shifted[cell] = L[cell - t]
                Ls_real = torch.roll(L_real, shifts=(tx, ty, tz), dims=(0, 1, 2))
                Ls_imag = torch.roll(L_imag, shifts=(tx, ty, tz), dims=(0, 1, 2))
                s = (R_real * Ls_real - R_imag * Ls_imag
                     - R_real * Ls_imag - R_imag * Ls_real).sum()
                out[tx, ty, tz] = s
    return out
