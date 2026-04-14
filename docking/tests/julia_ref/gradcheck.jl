# A-4: compare the hand-written `rrule(docking_score_elec)` gradients against
# central finite differences for a few representative parameters.
#
# Parameters are (α, iface_ij[144], β, charge_score[11]). Finite differences
# over all 157 would be expensive; we sample: α, β, iface_ij[1,13,72], and
# charge_score[1,6,11].
#
# Tolerance is 1e-3 relative (loose) because the score values are ~10^3 and
# gradients are integrated over many grid cells.

using Printf
using Statistics
using LinearAlgebra

using MDToolbox, Flux, ChainRulesCore, CUDA, ProgressMeter

const PROJECT_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const N_POSE = 5
const ALPHA = 0.01
const BETA = 3.0
const HALPHA = 1e-5
const HBETA = 1e-4
const HIFACE = 1e-4
const HCHARGE = 1e-4

include(joinpath(PROJECT_ROOT, "docking_canonical.jl"))

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
    receptor = set_charge(receptor)
    ligands  = set_charge(ligands)
    return receptor, ligands
end

receptor, ligands = setup_1kxq(N_POSE)

alpha = ALPHA
beta = BETA
iface_flat = reshape(MDToolbox.get_iface_ij(), :) |> Array{Float64}
charge = get_charge_score()

# Loss = sum of docking_score_elec over all poses
lossfn(a, ifs, b, chs) = sum(docking_score_elec(receptor, ligands, a, ifs, b, chs))

# AD gradients via the custom rrule
println("=== rrule-based gradients (via Flux.gradient) ===")
ad_grads = gradient(lossfn, alpha, iface_flat, beta, charge)
da_ad, difs_ad, db_ad, dchs_ad = ad_grads
@printf("  dL/dα         = %+.6e\n", da_ad)
@printf("  dL/dβ         = %+.6e\n", db_ad)
@printf("  dL/diface[1]  = %+.6e   (144-vec)\n", difs_ad[1])
@printf("  dL/diface[13] = %+.6e\n", difs_ad[13])
@printf("  dL/diface[72] = %+.6e\n", difs_ad[72])
@printf("  dL/dchs[1]    = %+.6e   (11-vec)\n", dchs_ad[1])
@printf("  dL/dchs[6]    = %+.6e\n", dchs_ad[6])
@printf("  dL/dchs[11]   = %+.6e\n", dchs_ad[11])

function fd(f, h)
    return (f(h) - f(-h)) / (2h)
end

println("\n=== Finite differences (central, h as indicated) ===")

# α
da_fd = fd(h -> lossfn(alpha + h, iface_flat, beta, charge), HALPHA)
@printf("  dL/dα (fd,h=%.0e) = %+.6e   rel_err vs AD = %.3e\n",
        HALPHA, da_fd, abs(da_fd - da_ad) / max(abs(da_fd), 1e-12))

# β
db_fd = fd(h -> lossfn(alpha, iface_flat, beta + h, charge), HBETA)
@printf("  dL/dβ (fd,h=%.0e) = %+.6e   rel_err vs AD = %.3e\n",
        HBETA, db_fd, abs(db_fd - db_ad) / max(abs(db_fd), 1e-12))

# iface[k] for k ∈ {1, 13, 72}
function perturb_iface(k, h)
    v = copy(iface_flat)
    v[k] += h
    return v
end
for k in (1, 13, 72)
    dk_fd = fd(h -> lossfn(alpha, perturb_iface(k, h), beta, charge), HIFACE)
    @printf("  dL/diface[%d] (fd,h=%.0e) = %+.6e   rel_err vs AD = %.3e\n",
            k, HIFACE, dk_fd, abs(dk_fd - difs_ad[k]) / max(abs(dk_fd), 1e-12))
end

# charge[k] for k ∈ {1, 6, 11}
function perturb_chs(k, h)
    v = copy(charge)
    v[k] += h
    return v
end
for k in (1, 6, 11)
    dk_fd = fd(h -> lossfn(alpha, iface_flat, beta, perturb_chs(k, h)), HCHARGE)
    @printf("  dL/dchs[%d] (fd,h=%.0e) = %+.6e   rel_err vs AD = %.3e\n",
            k, HCHARGE, dk_fd, abs(dk_fd - dchs_ad[k]) / max(abs(dk_fd), 1e-12))
end

println("\nDone.")
