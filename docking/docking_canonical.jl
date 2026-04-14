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
