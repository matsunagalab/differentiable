# B-0: Generate HDF5 reference outputs for the PyTorch TDD port.
#
# Layout: tests/refs/1KXQ/phaseN_<name>.h5
#
# Each file contains one HDF5 group named after the function, with datasets
# for the input arguments (prefixed `in_`) and the output (prefixed `out_`).
# Python-side tests read the inputs, re-run the PyTorch implementation, and
# assert numerical closeness with the Julia output.
#
# Run from docking/:
#   julia tests/julia_ref/generate_refs.jl
#
# Takes ~30 seconds on 10 poses of 1KXQ.

using Printf
using HDF5
using MDToolbox, Flux, ChainRulesCore, CUDA, ProgressMeter

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const REF_ROOT = joinpath(PROJECT_ROOT, "tests", "refs", "1KXQ")
const N_POSE = 10
const ALPHA = 0.01
const BETA = 3.0

include(joinpath(PROJECT_ROOT, "docking_canonical.jl"))

mkpath(REF_ROOT)

# ------------------------------------------------------------------
# Setup: load raw PDB, run set_radius / compute_sasa / set_atomtype_id /
# mass-init / set_charge — same as the training notebook does.
# ------------------------------------------------------------------
println(">>> Loading 1KXQ decoys (first $(N_POSE) poses)")

function load_decoys(n_pose::Int)
    pdb = mdload(joinpath(PROJECT_ROOT, "protein/1KXQ/complex.1.pdb"))
    rec = pdb[1:1, 1:3908]
    lig = pdb[1:1, 3909:end]
    for i in 2:n_pose
        pdb_i = mdload(joinpath(PROJECT_ROOT, "protein/1KXQ/complex.$(i).pdb"))
        lig = [lig; pdb_i[1:1, 3909:end]]
    end
    return rec, lig
end

receptor, ligands = load_decoys(N_POSE)

# Raw coordinates before any preprocessing (for io test)
rec_xyz_raw = Array{Float64}(receptor.xyz)
lig_xyz_raw = Array{Float64}(ligands.xyz)

# Apply the standard preprocessing.
receptor = MDToolbox.set_radius(receptor)
ligands  = MDToolbox.set_radius(ligands)
receptor = MDToolbox.compute_sasa(receptor)
ligands  = MDToolbox.compute_sasa(ligands)
receptor = MDToolbox.set_atomtype_id(receptor)
ligands  = MDToolbox.set_atomtype_id(ligands)
iface_score = MDToolbox.get_iface_ij()
receptor = TrjArray(receptor, mass=iface_score[receptor.atomtype_id])
ligands  = TrjArray(ligands,  mass=iface_score[ligands.atomtype_id])
receptor = set_charge(receptor)
ligands  = set_charge(ligands)

println("   receptor: $(receptor.natom) atoms, ligand: $(ligands.natom) atoms, $(ligands.nframe) frames")

# ------------------------------------------------------------------
# Helper: write an HDF5 group with attrs + datasets.
# ------------------------------------------------------------------
function writeh5(path::AbstractString, datasets::Dict)
    h5open(path, "w") do f
        for (k, v) in datasets
            f[String(k)] = v
        end
    end
    println("   wrote $(path)")
end

# ------------------------------------------------------------------
# Phase 1 — pure lookup tables & leaf functions.
# ------------------------------------------------------------------
println(">>> Phase 1 references")

writeh5(joinpath(REF_ROOT, "phase1_tables.h5"),
    Dict(
        "ace_score"     => get_acescore(),
        "iface_ij"      => Array{Float64}(iface_score),
        "iface_ij_flat" => reshape(Array{Float64}(iface_score), :),
        "charge_score"  => get_charge_score(),
    ))

let n = 960
    pts = golden_section_spiral(n)
    writeh5(joinpath(REF_ROOT, "phase1_spiral.h5"),
        Dict(
            "npoint" => n,
            "points" => pts,
        ))
end

# set_atomtype_id / set_radius / set_charge: raw PDB -> processed arrays.
# We record for the receptor of 1KXQ (first frame only).
writeh5(joinpath(REF_ROOT, "phase1_receptor_tables.h5"),
    Dict(
        "atomname"    => String.(receptor.atomname),
        "resname"     => String.(receptor.resname),
        "xyz_raw"     => rec_xyz_raw,
        "atomtype_id" => Array{Int64}(receptor.atomtype_id),
        "radius"      => Array{Float64}(receptor.radius),
        "charge"      => Array{Int64}(receptor.charge),
        "mass"        => Array{Float64}(receptor.mass),
    ))

writeh5(joinpath(REF_ROOT, "phase1_ligand_tables.h5"),
    Dict(
        "atomname"    => String.(ligands.atomname),
        "resname"     => String.(ligands.resname),
        "xyz_raw"     => lig_xyz_raw,
        "atomtype_id" => Array{Int64}(ligands.atomtype_id),
        "radius"      => Array{Float64}(ligands.radius),
        "charge"      => Array{Int64}(ligands.charge),
        "mass"        => Array{Float64}(ligands.mass),
    ))

# rotate!: small synthetic case because the full ligand rotation is only
# exercised by compute_docking_score_with_fft / docking.
let T = Float64
    x = T[1.0, 0.0, 0.0, 1.0]
    y = T[0.0, 1.0, 0.0, 1.0]
    z = T[0.0, 0.0, 1.0, 1.0]
    x_in, y_in, z_in = copy(x), copy(y), copy(z)
    q = T[0.5, 0.5, 0.5, 0.5]  # 120° about (1,1,1)
    rotate!(x, y, z, q)
    writeh5(joinpath(REF_ROOT, "phase1_rotate.h5"),
        Dict(
            "q"      => q,
            "x_in"   => x_in,
            "y_in"   => y_in,
            "z_in"   => z_in,
            "x_out"  => x,
            "y_out"  => y,
            "z_out"  => z,
        ))
end

# ------------------------------------------------------------------
# Phase 2 — SASA and grid generation.
# ------------------------------------------------------------------
println(">>> Phase 2 references")

# SASA was already computed above for both receptor and ligand. Capture.
writeh5(joinpath(REF_ROOT, "phase2_sasa.h5"),
    Dict(
        "receptor_xyz"    => rec_xyz_raw,
        "receptor_radius" => Array{Float64}(receptor.radius),
        "receptor_sasa"   => Array{Float64}(receptor.sasa),
        "ligand_xyz"      => lig_xyz_raw,
        "ligand_radius"   => Array{Float64}(ligands.radius),
        "ligand_sasa"     => Array{Float64}(ligands.sasa),
    ))

# generate_grid at spacing=3.0 (what docking_score_elec uses).
#
# Julia's generate_grid deep-copies the inputs and then applies `decenter!`
# to the receptor and `orient!` (PCA rotation) to the ligand BEFORE computing
# the grid bounds. Our Python generate_grid takes already-prepped coords, so
# we mirror the internal prep here and store the post-prep coordinates — the
# Python side then reads those and computes an identical grid.
let spacing = 3.0
    rec2 = deepcopy(receptor); decenter!(rec2)
    lig2 = deepcopy(ligands);  orient!(lig2)

    grid_real, grid_imag, x_grid, y_grid, z_grid = generate_grid(rec2, lig2, spacing=spacing)
    writeh5(joinpath(REF_ROOT, "phase2_grid.h5"),
        Dict(
            "spacing"             => spacing,
            "receptor_xyz_prep"   => Array{Float64}(rec2.xyz),
            "ligand_xyz_prep"     => Array{Float64}(lig2.xyz),
            "x_grid"              => Array{Float64}(x_grid),
            "y_grid"              => Array{Float64}(y_grid),
            "z_grid"              => Array{Float64}(z_grid),
            "grid_shape"          => collect(size(grid_real)),
        ))
end

# ------------------------------------------------------------------
# Phase 3 — scatter / spread operations on a small synthetic grid.
# ------------------------------------------------------------------
println(">>> Phase 3 references")

let T = Float64
    # Small reproducible 3D grid for unit-testing scatter ops.
    spacing = 1.0
    x_grid = collect(T, range(0.0, 10.0, step=spacing))
    y_grid = collect(T, range(0.0, 8.0, step=spacing))
    z_grid = collect(T, range(0.0, 6.0, step=spacing))
    nx, ny, nz = length(x_grid), length(y_grid), length(z_grid)

    # A handful of atoms with non-trivial positions/weights/rcut.
    x = T[2.3, 5.7, 8.1, 0.5, 4.0]
    y = T[1.4, 6.3, 2.9, 7.2, 3.1]
    z = T[0.8, 3.6, 2.4, 4.5, 1.2]
    weight = T[1.0, 2.0, -0.5, 0.3, 0.7]
    rcut = T[2.0, 1.5, 1.2, 2.5, 1.8]

    # spread_nearest_add!
    grid = zeros(T, nx, ny, nz)
    spread_nearest_add!(grid, x, y, z, x_grid, y_grid, z_grid, weight)
    grid_nearest_add = copy(grid)

    # spread_nearest_substitute!
    grid .= zero(T)
    spread_nearest_substitute!(grid, x, y, z, x_grid, y_grid, z_grid, weight)
    grid_nearest_sub = copy(grid)

    # spread_neighbors_add!
    grid .= zero(T)
    spread_neighbors_add!(grid, x, y, z, x_grid, y_grid, z_grid, weight, rcut)
    grid_neigh_add = copy(grid)

    # spread_neighbors_substitute!
    # Use uniform weight so the test is deterministic on all backends —
    # MPS's `index_put_(accumulate=False)` has non-deterministic tie-break
    # for duplicate indices. Production usage always passes a uniform
    # weight within one call (assign_sc_* all do `weight .= scalar`), so
    # the uniform-weight case is what actually matters.
    weight_uniform = similar(weight); weight_uniform .= T(3.5)
    grid .= zero(T)
    spread_neighbors_substitute!(grid, x, y, z, x_grid, y_grid, z_grid, weight_uniform, rcut)
    grid_neigh_sub = copy(grid)

    # calculate_distance!
    grid .= zero(T)
    calculate_distance!(grid, x, y, z, x_grid, y_grid, z_grid, weight, rcut)
    grid_calc_dist = copy(grid)

    writeh5(joinpath(REF_ROOT, "phase3_spread.h5"),
        Dict(
            "spacing"        => spacing,
            "x_grid"         => x_grid,
            "y_grid"         => y_grid,
            "z_grid"         => z_grid,
            "x"              => x,
            "y"              => y,
            "z"              => z,
            "weight"         => weight,
            "rcut"           => rcut,
            "nearest_add"    => grid_nearest_add,
            "nearest_sub"    => grid_nearest_sub,
            "neigh_add"      => grid_neigh_add,
            "neigh_sub"      => grid_neigh_sub,
            "neigh_sub_weight" => T(3.5),
            "calc_dist"      => grid_calc_dist,
        ))
end

# ------------------------------------------------------------------
# Phase 4 — assign_* functions and Phase 5 — full docking_score.
#
# We run docking_score_elec on a subset of poses and capture the per-pose
# scalar scores. Intermediate grids are harder to capture cheaply without
# monkey-patching the function; we punt on intermediate-grid captures for
# now and rely on end-to-end scalar matching as the TDD gate.
# ------------------------------------------------------------------
println(">>> Phase 5 references (full docking_score_elec forward + inputs)")

iface_flat = reshape(iface_score, :) |> Array{Float64}
charge_arr = get_charge_score()

# Mirror what docking_score_elec does internally: decenter! both, then
# run generate_grid on the same inputs (which internally re-orients the
# ligand). We save the post-decenter receptor/ligand as the Python inputs
# and the post-orient ligand (frame 1 only) separately so the Python side
# can reconstruct the grid without porting orient!.
function prepare_for_scoring(rec0, lig0)
    rec = deepcopy(rec0); decenter!(rec)
    lig = deepcopy(lig0); decenter!(lig)
    lig_for_grid = deepcopy(lig); orient!(lig_for_grid)
    return rec, lig, lig_for_grid
end

rec_prep, lig_prep, lig_for_grid = prepare_for_scoring(receptor, ligands)

# Ligand xyz reshaped as (nframe, natom, 3) — notebook stores
# ta.xyz as (nframe, 3*natom) row-major, so split triples.
function xyz_matrix(ta)
    # ta.xyz is (nframe, 3*natom); reshape to (nframe, natom, 3)
    xyz = Array{Float64}(ta.xyz)
    nframe, ntriple = size(xyz)
    natom = ntriple ÷ 3
    out = Array{Float64}(undef, nframe, natom, 3)
    for f in 1:nframe, a in 1:natom
        out[f, a, 1] = xyz[f, 3*(a-1)+1]
        out[f, a, 2] = xyz[f, 3*(a-1)+2]
        out[f, a, 3] = xyz[f, 3*(a-1)+3]
    end
    return out
end

scores_dse = docking_score_elec(receptor, ligands, ALPHA, iface_flat, BETA, charge_arr)
scores_ds  = docking_score(receptor, ligands, ALPHA, iface_flat)

writeh5(joinpath(REF_ROOT, "phase5_scores.h5"),
    Dict(
        # constants
        "alpha"            => ALPHA,
        "beta"             => BETA,
        "iface_ij_flat"    => iface_flat,
        "charge_score"     => charge_arr,
        "n_pose"           => N_POSE,
        # prepared inputs (post-decenter for scoring; oriented ligand for grid)
        "rec_xyz"          => xyz_matrix(rec_prep)[1, :, :],  # (N_rec, 3)
        "rec_radius"       => Array{Float64}(rec_prep.radius),
        "rec_sasa"         => Array{Float64}(rec_prep.sasa),
        "rec_atomtype_id"  => Array{Int64}(rec_prep.atomtype_id),
        "rec_charge_id"    => Array{Int64}(rec_prep.charge),
        "lig_xyz"          => xyz_matrix(lig_prep),            # (F, N_lig, 3)
        "lig_xyz_for_grid" => xyz_matrix(lig_for_grid)[1, :, :],  # for grid bounds
        "lig_radius"       => Array{Float64}(lig_prep.radius),
        "lig_sasa"         => Array{Float64}(lig_prep.sasa),
        "lig_atomtype_id"  => Array{Int64}(lig_prep.atomtype_id),
        "lig_charge_id"    => Array{Int64}(lig_prep.charge),
        # outputs
        "score_elec_total" => Array{Float64}(scores_dse),
        "score_noelec"     => Array{Float64}(scores_ds),
    ))

println(">>> All references written to $(REF_ROOT)")
