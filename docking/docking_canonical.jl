# docking_canonical.jl
#
# Canonical Julia source for the differentiable ZDOCK pipeline as used in
# `train_param-apart.ipynb` (the final training notebook of the master thesis
# by 柿沼孝紀). This file composes two sources:
#
#   1. docking.jl             — the MDToolbox-style baseline (kept verbatim)
#   2. docking_canonical_overrides.jl
#                             — function definitions from the notebook that
#                               OVERRIDE and EXTEND docking.jl; this is the
#                               code that actually ran for the thesis results
#
# Known bugs in the notebook's code are fixed here (see PORT_PLAN.md,
# "発見されたバグ一覧"). Each fix lives as a self-contained edit inside
# docking_canonical_overrides.jl so git blame shows which bug each line came
# from.
#
# To reproduce the notebook behavior bit-for-bit (pre-fix), rerun
# `tools/extract_notebook.py` and skip the fix edits.

include(joinpath(@__DIR__, "docking.jl"))
include(joinpath(@__DIR__, "docking_canonical_overrides.jl"))

# ---------------------------------------------------------------------------
# Additional bug fixes on top of the notebook + docking.jl code.
# ---------------------------------------------------------------------------

# B8. `docking.jl` set_radius (lines 828–858) uses `match(r"H.*", atomname)`
# which finds H anywhere in the name — classifying physically-oxygen atoms
# like "OH" (TYR hydroxyl) as hydrogen (radius 1.20) instead of oxygen
# (radius 1.52). Same for "NH1"/"NH2" (should be N, not H). The correct
# rule is: the element is the first non-digit character of the atom name
# (PDB convention — e.g. "1HG1" → H, "NH1" → N, "OH" → O).
#
# We patch `MDToolbox.set_radius` in place so existing call sites that
# qualify the function (`MDToolbox.set_radius(receptor)` in the notebook
# and generate_refs.jl) pick up the fix.
@eval MDToolbox function set_radius(ta::TrjArray{T,U}) where {T,U}
    radius_dict = Dict("H" => T(1.20),
                       "C" => T(1.70),
                       "N" => T(1.55),
                       "O" => T(1.52),
                       "S" => T(1.80))
    radius = Array{T}(undef, ta.natom)
    for iatom in 1:ta.natom
        name = ta.atomname[iatom]
        idx = 1
        while idx <= length(name) && isdigit(name[idx])
            idx += 1
        end
        element = idx <= length(name) ? string(name[idx]) : ""
        if !haskey(radius_dict, element)
            error("set_radius: unknown element in atom name $(name)")
        end
        radius[iatom] = radius_dict[element]
    end
    return TrjArray(ta, radius=radius)
end

# ---------------------------------------------------------------------------
# B10 / B11 / B12 / B13 fix: physically correct Coulombic electrostatics.
#
# The notebook's assign_Re_potential! / assign_Li_potential! compute
# `Σq / Σr` (a count-over-distance-sum quotient that is not a physical
# quantity), the ligand side is stored with a wrong sign, pairs are
# restricted to same-atomtype groups, and receptor core cells are not
# zeroed. Chen & Weng 2002 (p284) and Chen et al. 2003 (Eq 2) are
# unambiguous:
#
#   V_rec(r) = Σⱼ qⱼ / |r − rⱼ|            (Coulombic potential)
#   Im[L](cell) = −q_atom  at nearest cell of each ligand atom
#   score_ELEC  = β · Σ_cells V_rec(cell) · Im[L](cell)
#                = β · Σ_lig q_lig · V_rec(r_lig)    (= β × Coulomb energy)
#
# We expose a separate `docking_score_elec_coulomb` so the original
# `docking_score_elec` (legacy, notebook-faithful) remains available for
# thesis-reproduction work.
# ---------------------------------------------------------------------------

function spread_neighbors_coulomb!(
    grid::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    charges::AbstractVector{T}, rcut::T; d_floor::T=T(1e-6)
) where {T}
    natom = length(x)
    nx, ny, nz = size(grid)
    for iatom in 1:natom
        q = charges[iatom]
        r2 = rcut * rcut
        for ix in 1:nx
            dx = x[iatom] - x_grid[ix]
            if abs(dx) > rcut
                continue
            end
            for iy in 1:ny
                dy = y[iatom] - y_grid[iy]
                if abs(dy) > rcut
                    continue
                end
                for iz in 1:nz
                    dz = z[iatom] - z_grid[iz]
                    if abs(dz) > rcut
                        continue
                    end
                    d2 = dx*dx + dy*dy + dz*dz
                    if d2 < r2
                        d = sqrt(d2)
                        grid[ix, iy, iz] += q / max(d, d_floor)
                    end
                end
            end
        end
    end
    return nothing
end

"""
    docking_score_elec_coulomb(receptor, ligands, alpha, iface_ij, beta, charge_score)

Physically-correct ZDOCK-style docking score with Coulombic electrostatics
per Chen 2002/2003. Same shape/IFACE terms as `docking_score_elec`, but the
ELEC term is the actual Coulomb interaction energy Σ qᵢ qⱼ / rᵢⱼ instead of
the notebook's `Σq/Σr` pseudo-quantity.

receptor.charge / ligands.charge are interpreted as 1-based IDs into the
11-entry `charge_score` vector — a stopgap for a full CHARMM19 per-atom
partial-charge table (B14). Receptor-interior grid cells are zeroed so the
Coulomb sum only acts across the receptor surface into free space.
"""
function docking_score_elec_coulomb(receptor_org::TrjArray{T,U},
        ligands_org::TrjArray{T,U}, alpha::T, iface_ij::AbstractArray{T},
        beta::T, charge_score::AbstractArray{T}) where {T,U}
    spacing = T(3.0)
    rcut_elec = T(8.0)
    receptor = deepcopy(receptor_org)
    ligands  = deepcopy(ligands_org)

    decenter!(receptor)
    decenter!(ligands)

    grid_real, grid_imag, x_grid, y_grid, z_grid = generate_grid(receptor, ligands, spacing=spacing)

    com = centerofmass(receptor)
    receptor.xyz[:, 1:3:end] .= receptor.xyz[:, 1:3:end] .- com.xyz[:, 1:1]
    receptor.xyz[:, 2:3:end] .= receptor.xyz[:, 2:3:end] .- com.xyz[:, 2:2]
    receptor.xyz[:, 3:3:end] .= receptor.xyz[:, 3:3:end] .- com.xyz[:, 3:3]
    ligands.xyz[:, 1:3:end] .= ligands.xyz[:, 1:3:end] .- com.xyz[:, 1:1]
    ligands.xyz[:, 2:3:end] .= ligands.xyz[:, 2:3:end] .- com.xyz[:, 2:2]
    ligands.xyz[:, 3:3:end] .= ligands.xyz[:, 3:3:end] .- com.xyz[:, 3:3]

    x = receptor.xyz[1, 1:3:end]
    y = receptor.xyz[1, 2:3:end]
    z = receptor.xyz[1, 3:3:end]
    id_surface = receptor.sasa .> T(1.0)

    # -- SC receptor (same as docking_score_elec) --
    assign_sc_receptor_plus!(grid_real, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
    assign_sc_receptor_minus!(grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
    grid_sc_receptor = grid_real .+ im .* grid_imag

    # -- Coulombic V_rec: spread Σqⱼ/|r−rⱼ| on grid, zero inside receptor --
    # `receptor.charge` may be stored as Float64 in TrjArray; cast to Int for
    # the charge_score LUT index.
    rec_partial_q = charge_score[Int.(receptor.charge)]   # (N_rec,) per-atom charge
    V_rec = similar(grid_real)
    V_rec .= zero(T)
    spread_neighbors_coulomb!(V_rec, x, y, z, x_grid, y_grid, z_grid,
                              rec_partial_q, rcut_elec)
    # Zero V_rec in cells that lie inside the receptor (SC shape nonzero).
    open_mask = (grid_real .== zero(T)) .& (grid_imag .== zero(T))
    V_rec .= V_rec .* T.(open_mask)

    # -- Ligand SC and per-frame scoring loop --
    x = ligands.xyz[1, 1:3:end]
    y = ligands.xyz[1, 2:3:end]
    z = ligands.xyz[1, 3:3:end]

    x2 = receptor.xyz[1, 1:3:end]
    y2 = receptor.xyz[1, 2:3:end]
    z2 = receptor.xyz[1, 3:3:end]

    id_surface_lig = ligands.sasa .> T(1.0)

    grid_sc_ligand = deepcopy(grid_sc_receptor)
    score_sc    = similar(grid_real, ligands.nframe); score_sc    .= zero(T)
    score_iface = similar(grid_real, ligands.nframe); score_iface .= zero(T)
    score_elec  = similar(grid_real, ligands.nframe); score_elec  .= zero(T)
    score_total = similar(grid_real, ligands.nframe)
    grid_Q_L    = similar(grid_real)                  # ligand charge grid

    @showprogress for iframe = 1:ligands.nframe
        x .= ligands.xyz[iframe, 1:3:end]
        y .= ligands.xyz[iframe, 2:3:end]
        z .= ligands.xyz[iframe, 3:3:end]

        # SC ligand
        assign_sc_ligand_plus!(grid_real, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface_lig)
        assign_sc_ligand_minus!(grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface_lig)
        grid_sc_ligand .= grid_real .+ im .* grid_imag
        multi = grid_sc_receptor .* grid_sc_ligand
        score_sc[iframe] = sum(real.(multi)) - sum(imag.(multi))

        # IFACE (identical to docking_score_elec)
        for i in 1:12
            idx = ligands.atomtype_id .== i
            if any(idx)
                assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
                for j in 1:12
                    k = 12 * (j - 1) + i
                    receptor.mass .= iface_ij[k]
                    idx2 = receptor.atomtype_id .== j
                    if any(idx2)
                        assign_Rij!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])
                        score_iface[iframe] += sum(grid_real .* grid_imag)
                    end
                end
            end
        end

        # Coulombic ELEC: Q_L[cell] = Σ q_lig_atom at nearest cell.
        # score_elec[frame] = Σ_cells V_rec × Q_L = Σ_lig q_lig × V_rec(r_lig)
        grid_Q_L .= zero(T)
        lig_partial_q = charge_score[Int.(ligands.charge)]  # (N_lig,)
        spread_nearest_add!(grid_Q_L, x, y, z, x_grid, y_grid, z_grid, lig_partial_q)
        score_elec[iframe] = sum(V_rec .* grid_Q_L)

        score_total[iframe] = alpha * score_sc[iframe] + score_iface[iframe] + beta * score_elec[iframe]
    end

    return score_total
end
