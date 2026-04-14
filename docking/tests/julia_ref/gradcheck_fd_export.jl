# Save Julia-side central-difference gradients of docking_score_elec to
# HDF5 so the PyTorch side can test its autograd result against Julia
# FD (not Julia rrule — the hand-written rrule has bugs B6/B7, see
# gradcheck_report.md).
#
# Exports:
#   refs/1KXQ/phase6_fd_grads.h5
#     alpha_fd, beta_fd  (scalars)
#     iface_fd           (144,) — central FD per element
#     charge_fd          (11,)  — central FD per element
#     loss_value         scalar — loss at nominal parameters (sum of scores)
#
# Run from docking/:
#   julia tests/julia_ref/gradcheck_fd_export.jl

using HDF5
using Printf
using MDToolbox, Flux, ChainRulesCore, CUDA, ProgressMeter

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const REF_ROOT = joinpath(PROJECT_ROOT, "tests", "refs", "1KXQ")
const N_POSE = 5       # smaller = faster FD sweep
const ALPHA = 0.01
const BETA = 3.0
const HA = 1e-5
const HB = 1e-4
const HI = 1e-4
const HC = 1e-4

include(joinpath(PROJECT_ROOT, "docking_canonical.jl"))

function setup_1kxq(n_pose::Int)
    pdb = mdload(joinpath(PROJECT_ROOT, "protein/1KXQ/complex.1.pdb"))
    rec = pdb[1:1, 1:3908]
    lig = pdb[1:1, 3909:end]
    for i in 2:n_pose
        pdb_i = mdload(joinpath(PROJECT_ROOT, "protein/1KXQ/complex.$(i).pdb"))
        lig = [lig; pdb_i[1:1, 3909:end]]
    end
    rec = MDToolbox.set_radius(rec); lig = MDToolbox.set_radius(lig)
    rec = MDToolbox.compute_sasa(rec); lig = MDToolbox.compute_sasa(lig)
    rec = MDToolbox.set_atomtype_id(rec); lig = MDToolbox.set_atomtype_id(lig)
    iface_score = MDToolbox.get_iface_ij()
    rec = TrjArray(rec, mass=iface_score[rec.atomtype_id])
    lig = TrjArray(lig, mass=iface_score[lig.atomtype_id])
    rec = set_charge(rec); lig = set_charge(lig)
    return rec, lig
end

receptor, ligands = setup_1kxq(N_POSE)
alpha = ALPHA; beta = BETA
iface_flat = reshape(MDToolbox.get_iface_ij(), :) |> Array{Float64}
charge = get_charge_score()

lossfn(a, ifs, b, chs) = sum(docking_score_elec_coulomb(receptor, ligands, a, ifs, b, chs))

function fd(f, h)
    return (f(h) - f(-h)) / (2h)
end

loss0 = lossfn(alpha, iface_flat, beta, charge)
println("loss at nominal = $(loss0)")

println("--- dL/dα ---")
alpha_fd = fd(h -> lossfn(alpha + h, iface_flat, beta, charge), HA)
@printf("  %.6e\n", alpha_fd)

println("--- dL/dβ ---")
beta_fd = fd(h -> lossfn(alpha, iface_flat, beta + h, charge), HB)
@printf("  %.6e\n", beta_fd)

println("--- dL/diface (144 elements, ~3 min) ---")
iface_fd = zeros(144)
@showprogress for k in 1:144
    function pert(h)
        v = copy(iface_flat); v[k] += h
        return lossfn(alpha, v, beta, charge)
    end
    iface_fd[k] = fd(pert, HI)
end

println("--- dL/dcharge (11 elements) ---")
charge_fd = zeros(11)
@showprogress for l in 1:11
    function pert(h)
        v = copy(charge); v[l] += h
        return lossfn(alpha, iface_flat, beta, v)
    end
    charge_fd[l] = fd(pert, HC)
end

mkpath(REF_ROOT)
h5open(joinpath(REF_ROOT, "phase6_fd_grads.h5"), "w") do f
    f["alpha_fd"] = alpha_fd
    f["beta_fd"]  = beta_fd
    f["iface_fd"] = iface_fd
    f["charge_fd"] = charge_fd
    f["loss_value"] = loss0
    f["alpha"] = alpha
    f["beta"] = beta
    f["iface_ij_flat"] = iface_flat
    f["charge_score"] = charge
    f["n_pose"] = N_POSE
    f["halpha"] = HA
    f["hbeta"] = HB
    f["hiface"] = HI
    f["hcharge"] = HC
end
println("\nWrote $(joinpath(REF_ROOT, "phase6_fd_grads.h5"))")
