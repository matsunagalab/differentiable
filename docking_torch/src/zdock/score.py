"""Forward scoring functions (`docking_score`, `docking_score_elec`) ported
from `train_param-apart.ipynb` cell 4–5.

Design notes for GPU / MPS performance:

* The inner Julia loops iterate 12×12 = 144 times for the IFACE term and
  11 times for the ELEC term, each spread-ing a single atom type onto a
  fresh grid and doing one elementwise-multiply-and-sum. On the GPU each
  iteration is a tiny kernel launch; 144 launches per frame × nframe adds
  up.

  We replace the nested loops with **batched** spreads on grids of shape
  ``(n_type, nx, ny, nz)`` (one slab per atom type). The dot product
  across type pairs becomes a single ``L @ H.T`` matmul (12, 12) for
  IFACE; ELEC collapses to a single einsum.

* All backbone ops are ``torch`` tensor ops — no Python-level atom loops
  in the hot path. Preprocessing (atom-type ID, SASA, charge ID) is
  one-time and runs off-device through ``atomtypes``.

* ``docking_score_elec`` is fully autograd-safe: α, β, iface_ij, and
  charge_score are leaf tensors; ``score_total = α S_SC + S_IFACE +
  β S_ELEC`` is a linear combination of grid reductions, and every step
  (scatter_add, elementwise ops, reductions, complex multiply) has a
  built-in PyTorch VJP.

The caller is responsible for preparing inputs exactly as the Julia
`docking_score_elec` internally would:

  * ``receptor_xyz`` is already centred (``decenter!``).
  * ``ligand_xyz`` is already PCA-oriented (``orient!``) and centred.

This avoids porting MDToolbox's ``orient!`` at the cost of shifting the
burden to the preprocessing side (Julia `generate_refs.jl` does this).
"""

from __future__ import annotations

import math

import torch

from typing import Literal

from .geom import generate_grid, orient
from .atomtypes import iface_ij, partial_charge_per_atom
from .spread import (
    _neighbors_indices,
    _flat_index,
    _in_bounds,
    _nearest_cell_indices,
    spread_neighbors_coulomb,
)


ElecMode = Literal["coulomb", "legacy"]


# ---------------------------------------------------------------------------
# Grouped / batched spread helpers — a generalisation of spread.py that
# routes each atom's contribution into a per-type slab of a
# (G, nx, ny, nz) grid in a single scatter call.
# ---------------------------------------------------------------------------


def _grouped_spread_nearest_add(
    grid_batch: torch.Tensor,         # (G, nx, ny, nz) — zero-initialized
    xyz: torch.Tensor,                # (N, 3)
    group: torch.Tensor,              # (N,) int in [0, G)
    weights: torch.Tensor,            # (N,) same dtype as grid_batch
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    G, nx, ny, nz = grid_batch.shape
    ix, iy, iz = _nearest_cell_indices(xyz, x_grid, y_grid, z_grid)
    in_b = _in_bounds(ix, iy, iz, (nx, ny, nz))
    valid = in_b & (group >= 0) & (group < G)
    flat = (
        group[valid] * (nx * ny * nz)
        + ix[valid] * (ny * nz)
        + iy[valid] * nz
        + iz[valid]
    )
    grid_batch.view(-1).scatter_add_(0, flat, weights[valid])
    return grid_batch


def _grouped_spread_neighbors_add(
    grid_batch: torch.Tensor,
    xyz: torch.Tensor,
    group: torch.Tensor,
    weights: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    G, nx, ny, nz = grid_batch.shape
    flat_cell, _, atom_idx = _neighbors_indices(
        xyz, rcut, x_grid, y_grid, z_grid, (nx, ny, nz)
    )
    g = group[atom_idx]
    valid = (g >= 0) & (g < G)
    flat = g[valid] * (nx * ny * nz) + flat_cell[valid]
    contribs = weights[atom_idx[valid]]
    grid_batch.view(-1).scatter_add_(0, flat, contribs)
    return grid_batch


def _grouped_calculate_distance(
    grid_batch: torch.Tensor,
    xyz: torch.Tensor,
    group: torch.Tensor,
    rcut: torch.Tensor | float,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    z_grid: torch.Tensor,
) -> torch.Tensor:
    G, nx, ny, nz = grid_batch.shape
    flat_cell, d, atom_idx = _neighbors_indices(
        xyz, rcut, x_grid, y_grid, z_grid, (nx, ny, nz)
    )
    g = group[atom_idx]
    valid = (g >= 0) & (g < G)
    flat = g[valid] * (nx * ny * nz) + flat_cell[valid]
    grid_batch.view(-1).scatter_add_(0, flat, d[valid])
    return grid_batch


# ---------------------------------------------------------------------------
# SC (shape complementarity) assign helpers — direct ports of the
# assign_sc_*_plus!/minus! functions in train_param-apart.ipynb cell 4.
# ---------------------------------------------------------------------------


def _assign_sc_plus(
    grid: torch.Tensor,
    xyz: torch.Tensor,
    radius: torch.Tensor,
    id_surface: torch.Tensor,
    x_grid, y_grid, z_grid,
    *,
    receptor: bool,
) -> torch.Tensor:
    from .spread import spread_neighbors_substitute

    grid.zero_()
    surf = id_surface
    core = ~id_surface

    weight_s = torch.ones_like(radius[surf])
    weight_c = torch.ones_like(radius[core])

    if surf.any():
        if receptor:
            spread_neighbors_substitute(
                grid, xyz[surf], weight_s, radius[surf] + 3.4, x_grid, y_grid, z_grid
            )
        spread_neighbors_substitute(
            grid, xyz[surf], weight_s,
            radius[surf] * math.sqrt(0.8) if receptor else radius[surf],
            x_grid, y_grid, z_grid,
        )
    if core.any():
        spread_neighbors_substitute(
            grid, xyz[core], weight_c, radius[core] * math.sqrt(1.5), x_grid, y_grid, z_grid
        )
    return grid


def _assign_sc_minus(
    grid: torch.Tensor,
    xyz: torch.Tensor,
    radius: torch.Tensor,
    id_surface: torch.Tensor,
    x_grid, y_grid, z_grid,
    *,
    receptor: bool,
) -> torch.Tensor:
    from .spread import spread_neighbors_substitute

    grid.zero_()
    surf = id_surface
    core = ~id_surface

    if surf.any():
        if receptor:
            weight_s_1 = torch.full_like(radius[surf], 3.5)
            spread_neighbors_substitute(
                grid, xyz[surf], weight_s_1, radius[surf] + 3.4, x_grid, y_grid, z_grid
            )
        weight_s_2 = torch.full_like(radius[surf], 12.25)
        spread_neighbors_substitute(
            grid, xyz[surf], weight_s_2,
            radius[surf] * math.sqrt(0.8) if receptor else radius[surf],
            x_grid, y_grid, z_grid,
        )

    if core.any():
        weight_c = torch.full_like(radius[core], 12.25)
        spread_neighbors_substitute(
            grid, xyz[core], weight_c, radius[core] * math.sqrt(1.5), x_grid, y_grid, z_grid
        )
    return grid


# ---------------------------------------------------------------------------
# Main entry point: docking_score_elec.
# ---------------------------------------------------------------------------


def docking_score_elec(
    rec_xyz: torch.Tensor,              # (N_rec, 3) — already decentered
    rec_radius: torch.Tensor,           # (N_rec,)
    rec_sasa: torch.Tensor,             # (N_rec,)
    rec_atomtype_id: torch.Tensor,      # (N_rec,) int in [1, 12]
    rec_charge_id: torch.Tensor,        # (N_rec,) int in [1, 11]
    lig_xyz: torch.Tensor,              # (F, N_lig, 3) — each frame already oriented+decentered
    lig_radius: torch.Tensor,           # (N_lig,)
    lig_sasa: torch.Tensor,             # (N_lig,)
    lig_atomtype_id: torch.Tensor,      # (N_lig,)
    lig_charge_id: torch.Tensor,        # (N_lig,)
    alpha: torch.Tensor,                # scalar
    iface_ij_flat: torch.Tensor,        # (144,) — column-major of 12x12
    beta: torch.Tensor,                 # scalar
    charge_score: torch.Tensor,         # (11,)
    *,
    lig_xyz_for_grid: torch.Tensor | None = None,  # (N_lig, 3) post-orient
    spacing: float = 3.0,
    rcut_iface: float = 6.0,
    rcut_elec: float = 8.0,
    surface_threshold: float = 1.0,
    elec_mode: ElecMode = "coulomb",
) -> torch.Tensor:
    """Return a (F,) tensor of docking scores.

    Default `elec_mode="coulomb"` implements the physically-correct Chen 2002 /
    Chen 2003 ELEC: receptor generates a Coulombic potential V(r) = Σⱼ qⱼ / |r−rⱼ|
    (zeroed inside the receptor SC shape), ligand stores -q at the nearest grid
    cell of each atom, and `score_elec` accumulates V × (-q) ≡ Coulomb energy
    across all (lig-atom × rec-atom) pairs. β scales this sum in `score_total`.

    `elec_mode="legacy"` preserves the notebook's original (buggy) formulation
    that groups ELEC by atom-type and computes `Σq / Σr` instead of Σq/r. This
    matches the Julia reference before the B10/B11/B12/B13 fixes and exists for
    bit-exact reproduction of the master thesis numbers.

    SC + IFACE are unchanged between modes (they have no ELEC-specific bugs).
    """
    device = rec_xyz.device
    dtype = rec_xyz.dtype

    # Reshape iface_ij_flat (column-major 12×12) → (12, 12) matrix where
    # M[i, j] = iface_ij_flat[12*j + i]. Julia's k = 12*(j-1)+i maps to
    # Python index 12*j+i after 1-based → 0-based. The fortran-order view:
    iface_matrix = iface_ij_flat.view(12, 12).T  # (12, 12), M[i, j]

    # Julia's generate_grid applies `orient!` (PCA rotation) to the
    # ligand internally before computing grid bounds. We compute the same
    # rotation in Python via `orient()`, using the ligand IFACE values
    # as inertia weights (matching Julia's notebook preprocessing that
    # sets `ligands.mass = iface_score[atomtype_id]`). If the caller
    # overrides with `lig_xyz_for_grid`, use that directly (useful for
    # tests that pin to Julia's exact SVD sign choice).
    if lig_xyz_for_grid is not None:
        grid_bounds_lig = lig_xyz_for_grid
    else:
        iface_matrix_for_mass = iface_ij(device=device, dtype=dtype)
        lig_mass_weights = iface_matrix_for_mass[lig_atomtype_id - 1, 0]
        grid_bounds_lig = orient(lig_xyz[0], mass=lig_mass_weights)
    grid_real, grid_imag, x_grid, y_grid, z_grid = generate_grid(
        rec_xyz, grid_bounds_lig, spacing=spacing
    )
    nx, ny, nz = grid_real.shape

    # Precompute receptor SC slabs (real + imag parts of SC filter).
    rec_surf = rec_sasa > surface_threshold
    rec_sc_real = torch.zeros_like(grid_real)
    rec_sc_imag = torch.zeros_like(grid_imag)
    _assign_sc_plus(rec_sc_real, rec_xyz, rec_radius, rec_surf,
                    x_grid, y_grid, z_grid, receptor=True)
    _assign_sc_minus(rec_sc_imag, rec_xyz, rec_radius, rec_surf,
                     x_grid, y_grid, z_grid, receptor=True)

    # Precompute receptor IFACE contribution slabs H[j] for j in 1..12.
    # H[j] = Σ_atoms_of_type_j (within rcut=6 of cell) indicator. Weight 1.
    H = torch.zeros((12, nx, ny, nz), device=device, dtype=dtype)
    rec_group_iface = (rec_atomtype_id - 1).to(torch.long).clamp(0, 11)
    rec_weights_ones = torch.ones(rec_xyz.shape[0], device=device, dtype=dtype)
    _grouped_spread_neighbors_add(
        H, rec_xyz, rec_group_iface, rec_weights_ones, rcut_iface,
        x_grid, y_grid, z_grid,
    )

    # --- Receptor ELEC (mode-dependent) ---------------------------------

    if elec_mode == "coulomb":
        # Chen 2002 p284: V(r) = Σⱼ qⱼ / |r − rⱼ|. Zero out cells that fall
        # inside the receptor SC shape (Chen 2002 p284: "grid points in the
        # core of the receptor are assigned a value of 0 for the electric
        # potential"). `rec_sc_real > 0` covers both surface shell and core
        # per the SC encoding, so V_rec is populated only in open space.
        rec_partial_q = partial_charge_per_atom(rec_charge_id, charge_score)
        V_rec = torch.zeros((nx, ny, nz), device=device, dtype=dtype)
        spread_neighbors_coulomb(
            V_rec, rec_xyz, rec_partial_q, rcut_elec,
            x_grid, y_grid, z_grid,
        )
        open_space_mask = (rec_sc_real == 0) & (rec_sc_imag == 0)
        V_rec = V_rec * open_space_mask.to(dtype)
    else:  # elec_mode == "legacy"
        # Original notebook behaviour: group by atomtype_id (B9), compute
        # per-type `count / Σ√d` pseudo-potential (B10). Preserved for
        # reproducing thesis numbers bit-for-bit against the original Julia
        # reference.
        rec_in_charge = (rec_atomtype_id >= 1) & (rec_atomtype_id <= 11)
        rec_group_charge = torch.where(
            rec_in_charge, rec_atomtype_id - 1, torch.full_like(rec_atomtype_id, -1)
        ).to(torch.long)
        rec_xyz_c = rec_xyz[rec_in_charge]
        rec_group_c = rec_group_charge[rec_in_charge]
        U_num = torch.zeros((11, nx, ny, nz), device=device, dtype=dtype)
        U_den = torch.zeros((11, nx, ny, nz), device=device, dtype=dtype)
        _grouped_spread_nearest_add(
            U_num, rec_xyz_c, rec_group_c,
            torch.ones(rec_xyz_c.shape[0], device=device, dtype=dtype),
            x_grid, y_grid, z_grid,
        )
        _grouped_calculate_distance(
            U_den, rec_xyz_c, rec_group_c, rcut_elec,
            x_grid, y_grid, z_grid,
        )
        eps = torch.finfo(dtype).eps
        U = torch.where(U_den > 0, U_num / U_den.clamp(min=eps), torch.zeros_like(U_num))

    # Frame-batched scoring. The loop over frames in the Julia code is the
    # single largest source of GPU kernel launch overhead; we vectorize it
    # away by flattening (frame, atom) into a single atom dim and routing
    # each scatter into the correct per-frame slab with an extra frame
    # offset in the flat index.
    F = lig_xyz.shape[0]
    N_lig = lig_xyz.shape[1]
    lig_group_iface = (lig_atomtype_id - 1).to(torch.long).clamp(0, 11)
    # Same note as above: ELEC uses atomtype_id (1..11), not charge_id.
    lig_in_charge = (lig_atomtype_id >= 1) & (lig_atomtype_id <= 11)
    lig_group_charge = torch.where(
        lig_in_charge, lig_atomtype_id - 1, torch.full_like(lig_atomtype_id, -1)
    ).to(torch.long)
    lig_surf = lig_sasa > surface_threshold
    V = nx * ny * nz

    # SC ligand (batched per frame). We still use the three-layer
    # substitute spreads; rather than batching by frame here (which would
    # require a new grouped_spread_neighbors_substitute variant) we take
    # advantage of the fact that only 3 calls × F frames happen and pay
    # the launch overhead once per layer for all frames together.
    frame_arange = torch.arange(F, device=device)
    lig_surf_expanded = lig_surf.unsqueeze(0).expand(F, -1)      # (F, N)
    lig_radius_expanded = lig_radius.unsqueeze(0).expand(F, -1)  # (F, N)

    # Flattened per-atom arrays for batched scatter across frames.
    frame_idx_per_atom = frame_arange.unsqueeze(-1).expand(-1, N_lig).reshape(-1)
    lxyz_flat = lig_xyz.reshape(-1, 3)
    lig_radius_flat = lig_radius_expanded.reshape(-1)
    lig_group_iface_flat = lig_group_iface.unsqueeze(0).expand(F, -1).reshape(-1)
    lig_surf_flat = lig_surf_expanded.reshape(-1)
    lig_group_charge_flat = lig_group_charge.unsqueeze(0).expand(F, -1).reshape(-1)

    # --- SC ligand (complex) --------------------------------------------
    # For the substitute path, we do one "any-within-range" union mask per
    # layer, then overwrite. Write into a (F, nx, ny, nz) grid via
    # _grouped_spread_neighbors_add + clip-to-1 for union; this gives a
    # true union without relying on scatter's non-deterministic duplicate
    # handling.
    def sc_union(xyz_f, radii_f, group_frame, rcut_f, grid_shape):
        """Compute (F, nx, ny, nz) grid where cell is 1 if ANY of the
        given atoms (of each frame) has that cell within its rcut."""
        cnt = torch.zeros(grid_shape, device=device, dtype=dtype)
        _grouped_spread_neighbors_add(
            cnt, xyz_f, group_frame,
            torch.ones(xyz_f.shape[0], device=device, dtype=dtype),
            rcut_f, x_grid, y_grid, z_grid,
        )
        return (cnt > 0).to(dtype)

    surf_mask_flat = lig_surf_flat
    core_mask_flat = ~lig_surf_flat

    # lig_plus (real): 2 layers for ligand
    lig_sc_real = torch.zeros((F, nx, ny, nz), device=device, dtype=dtype)
    # layer 1 for ligand plus: surface atoms, rcut = radius
    surf_idx = surf_mask_flat.nonzero(as_tuple=True)[0]
    if surf_idx.numel() > 0:
        layer1 = sc_union(
            lxyz_flat[surf_idx], lig_radius_flat[surf_idx],
            frame_idx_per_atom[surf_idx], lig_radius_flat[surf_idx],
            (F, nx, ny, nz),
        )
        lig_sc_real = torch.maximum(lig_sc_real, layer1)
    # layer 2: core atoms, rcut = radius * sqrt(1.5)
    core_idx = core_mask_flat.nonzero(as_tuple=True)[0]
    if core_idx.numel() > 0:
        layer2 = sc_union(
            lxyz_flat[core_idx], lig_radius_flat[core_idx],
            frame_idx_per_atom[core_idx], lig_radius_flat[core_idx] * math.sqrt(1.5),
            (F, nx, ny, nz),
        )
        lig_sc_real = torch.maximum(lig_sc_real, layer2)

    # lig_minus (imag): 2 layers with weights 3.5 and 12.25
    lig_sc_imag = torch.zeros((F, nx, ny, nz), device=device, dtype=dtype)
    if surf_idx.numel() > 0:
        # Ligand minus does NOT apply the +3.4 pad (receptor-only in _assign_sc_minus).
        # First layer: surface, rcut = radius, weight 3.5
        lay1 = sc_union(
            lxyz_flat[surf_idx], lig_radius_flat[surf_idx],
            frame_idx_per_atom[surf_idx], lig_radius_flat[surf_idx],
            (F, nx, ny, nz),
        )
        lig_sc_imag = torch.where(lay1 > 0, lay1 * 3.5, lig_sc_imag)
    if core_idx.numel() > 0:
        lay2 = sc_union(
            lxyz_flat[core_idx], lig_radius_flat[core_idx],
            frame_idx_per_atom[core_idx], lig_radius_flat[core_idx] * math.sqrt(1.5),
            (F, nx, ny, nz),
        )
        lig_sc_imag = torch.where(lay2 > 0, lay2 * 12.25, lig_sc_imag)

    # Complex SC product per frame and reduce: score_sc[f] = Σ (real − imag)
    multi_real = rec_sc_real.unsqueeze(0) * lig_sc_real - rec_sc_imag.unsqueeze(0) * lig_sc_imag
    multi_imag = rec_sc_real.unsqueeze(0) * lig_sc_imag + rec_sc_imag.unsqueeze(0) * lig_sc_real
    score_sc = multi_real.reshape(F, -1).sum(-1) - multi_imag.reshape(F, -1).sum(-1)

    # --- IFACE: batched L[F, 12, V] -------------------------------------
    # Route each (frame, atom) → slab index = frame * 12 + type.
    L_count = torch.zeros((F * 12, nx, ny, nz), device=device, dtype=dtype)
    group_f12 = frame_idx_per_atom * 12 + lig_group_iface_flat
    _grouped_spread_nearest_add(
        L_count, lxyz_flat, group_f12,
        torch.ones(lxyz_flat.shape[0], device=device, dtype=dtype),
        x_grid, y_grid, z_grid,
    )
    L = (L_count.view(F, 12, nx, ny, nz) > 0).to(dtype)
    # T[f, i, j] = <L[f, i], H[j]>; batched bmm.
    T = torch.einsum("fiv,jv->fij", L.reshape(F, 12, V), H.reshape(12, V))
    score_iface = (iface_matrix.unsqueeze(0) * T).reshape(F, -1).sum(-1)

    # --- ELEC (mode-dependent forward) ----------------------------------

    if elec_mode == "coulomb":
        # Per-frame ligand charge grid Q_L: scatter +q_lig_atom at the
        # nearest cell. The paper's Im[L] = -q is absorbed into the final
        # sign below so that `score_elec` returns the Coulomb interaction
        # energy ΣΣ qᵢqⱼ / rᵢⱼ (negative for favorable, positive for
        # clashing). `score_total` then uses `+ β × score_elec` as in the
        # training notebook.
        lig_partial_q = partial_charge_per_atom(lig_charge_id, charge_score)
        lig_partial_q_flat = lig_partial_q.unsqueeze(0).expand(F, -1).reshape(-1)
        Q_L_flat = torch.zeros(F * nx * ny * nz, device=device, dtype=dtype)
        # frame_idx_per_atom already offsets by (nx*ny*nz) per frame when
        # we pack (frame * nx*ny*nz + cell) — reuse _grouped_spread_nearest_add
        # semantics by treating "frame" as the group.
        _grouped_spread_nearest_add(
            Q_L_flat.view(F, nx, ny, nz),
            lxyz_flat, frame_idx_per_atom, lig_partial_q_flat,
            x_grid, y_grid, z_grid,
        )
        Q_L = Q_L_flat.view(F, nx, ny, nz)
        # Coulomb interaction energy per frame:
        #   score_elec[f] = Σ_cells V_rec(cell) × Q_L[f, cell]
        #               = Σ_lig_atom q_lig × V_rec(r_lig_atom)
        #               = ΣΣ_ij q_i q_j / r_ij
        score_elec = (V_rec.unsqueeze(0) * Q_L).reshape(F, -1).sum(-1)
    else:  # elec_mode == "legacy"
        valid_flat = lig_group_charge_flat >= 0
        grp_f11 = frame_idx_per_atom * 11 + lig_group_charge_flat.clamp(min=0)
        grp_f11 = grp_f11[valid_flat]
        xyz_c_flat = lxyz_flat[valid_flat]
        V_count = torch.zeros((F * 11, nx, ny, nz), device=device, dtype=dtype)
        _grouped_spread_nearest_add(
            V_count, xyz_c_flat, grp_f11,
            torch.ones(xyz_c_flat.shape[0], device=device, dtype=dtype),
            x_grid, y_grid, z_grid,
        )
        V_grid = V_count.view(F, 11, nx, ny, nz)
        c = (V_grid.reshape(F, 11, V) * U.reshape(11, V).unsqueeze(0)).sum(-1)  # (F, 11)
        score_elec = (charge_score.pow(2).unsqueeze(0) * c).sum(-1)

    return alpha * score_sc + score_iface + beta * score_elec
