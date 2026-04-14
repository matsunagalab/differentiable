# Sanity check: compare score_sc, score_iface, score_elec distributions
# from the original (buggy) notebook code vs. the B1/B3-fixed canonical code.
#
# Loads 10 poses of 1KXQ decoys, runs docking_score_elec on both versions,
# and reports descriptive statistics. The fixed version should show score_elec
# with non-trivial magnitude; the buggy version has score_elec contaminated by
# stale IFACE grid state.

using Printf
using Statistics

using MDToolbox, Flux, ChainRulesCore, CUDA, ProgressMeter

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const N_POSE = 10
const ALPHA = 0.01
const BETA = 3.0

function setup_1kxq(n_pose::Int)
    pdb = mdload(joinpath(PROJECT_ROOT, "protein/1KXQ/complex.1.pdb"))
    receptor = pdb[1:1, 1:3908]
    ligands = pdb[1:1, 3909:end]
    for i in 2:n_pose
        pdb_i = mdload(joinpath(PROJECT_ROOT, "protein/1KXQ/complex.$(i).pdb"))
        ligands = [ligands; pdb_i[1:1, 3909:end]]
    end

    receptor = MDToolbox.set_radius(receptor)
    ligands  = MDToolbox.set_radius(ligands)
    receptor = MDToolbox.compute_sasa(receptor)
    ligands  = MDToolbox.compute_sasa(ligands)
    receptor = MDToolbox.set_atomtype_id(receptor)
    ligands  = MDToolbox.set_atomtype_id(ligands)
    iface_score = MDToolbox.get_iface_ij()
    receptor = TrjArray(receptor, mass=iface_score[receptor.atomtype_id])
    ligands  = TrjArray(ligands,  mass=iface_score[ligands.atomtype_id])
    return receptor, ligands, iface_score
end

# Load two flavors of the pipeline into separate modules so they don't clobber
# each other's method tables.
module Buggy
    using MDToolbox, Flux, ChainRulesCore, CUDA, ProgressMeter
    const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
    include(joinpath(ROOT, "docking.jl"))
    include(joinpath(ROOT, "docking_canonical_overrides_buggy.jl"))
end

module Fixed
    using MDToolbox, Flux, ChainRulesCore, CUDA, ProgressMeter
    const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
    include(joinpath(ROOT, "docking_canonical.jl"))
end

receptor, ligands, iface_score = setup_1kxq(N_POSE)
# set_charge uses the notebook-overridden version; we need per-module copies.
rec_buggy = Buggy.set_charge(receptor)
lig_buggy = Buggy.set_charge(ligands)
rec_fixed = Fixed.set_charge(receptor)
lig_fixed = Fixed.set_charge(ligands)

charge_buggy = Buggy.get_charge_score()
charge_fixed = Fixed.get_charge_score()

iface_flat = reshape(iface_score, :)

println("=== Running BUGGY pipeline ===")
scores_buggy = Buggy.docking_score_elec(rec_buggy, lig_buggy,
                                        ALPHA, iface_flat,
                                        BETA, charge_buggy)
println("\n=== Running FIXED pipeline ===")
scores_fixed = Fixed.docking_score_elec(rec_fixed, lig_fixed,
                                        ALPHA, iface_flat,
                                        BETA, charge_fixed)

function descstats(name, v)
    @printf("  %-20s  min=%+.4e  max=%+.4e  mean=%+.4e  std=%.4e\n",
            name, minimum(v), maximum(v), mean(v), std(v))
end

println()
println("=== score_total comparison (α=$(ALPHA), β=$(BETA), $(N_POSE) poses) ===")
descstats("BUGGY score_total", scores_buggy)
descstats("FIXED score_total", scores_fixed)
@printf("\n  max |Δ score_total|  = %.4e\n", maximum(abs.(scores_fixed .- scores_buggy)))

# Spot per-pose deltas
println("\n=== Per-pose score_total ===")
@printf("  %-6s  %-14s  %-14s  %-14s\n", "pose", "buggy", "fixed", "delta")
for i in 1:N_POSE
    @printf("  %-6d  %+.6e  %+.6e  %+.6e\n",
            i, scores_buggy[i], scores_fixed[i], scores_fixed[i] - scores_buggy[i])
end

println("\nDone.")
