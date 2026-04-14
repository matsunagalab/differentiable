# ============ cell 2 ============
function set_charge(ta::TrjArray{T,U}) where {T,U}
    charge = Array{Int64}(undef, ta.natom)
    
    atomname = deepcopy(ta.atomname)
    
    is_first = true
    
    for iatom = 1:ta.natom
        # ATOM TYPE "N"
        if ta.atomname[iatom] == "N"
            if is_first
                charge[iatom] = 1
                is_first = false
            else
                charge[iatom] = 11
            end

            # ATOM TYPE "O"
        elseif ta.resname[iatom] == "O"
            charge[iatom] = 10
        elseif ta.resname[iatom] == "OXT"
            charge[iatom] = 2

        elseif ta.resname[iatom] == "ARG" && atomname[iatom] == "NH1"
            charge[iatom] = 3
        elseif ta.resname[iatom] == "ARG" && atomname[iatom] == "NH2"
            charge[iatom] = 3
        elseif ta.resname[iatom] == "GLU" && atomname[iatom] == "OE1"
            charge[iatom] = 4
        elseif ta.resname[iatom] == "GLU" && atomname[iatom] == "OE2"
            charge[iatom] = 4
        elseif ta.resname[iatom] == "ASP" && atomname[iatom] == "OD1"
            charge[iatom] = 5
        elseif ta.resname[iatom] == "ASP" && atomname[iatom] == "OD2"
            charge[iatom] = 5
        elseif ta.resname[iatom] == "LYS" && atomname[iatom] == "NZ"
            charge[iatom] = 6
        elseif ta.resname[iatom] == "PRO" && atomname[iatom] == "N"
            charge[iatom] = 7

        else
            charge[iatom] = 8
        
        end
    end
    
    return TrjArray(ta, charge=charge)
end

function get_charge_score()
    charge_score = Array{Float64}(undef, 11)
    charge_score[1] = 1.0 #TERMINAL-N
    charge_score[2] = -1.0 #TERMINAL-O
    charge_score[3] = 0.5 #ARG,NH
    charge_score[4] = -0.5 #GLU,OE
    charge_score[5] = -0.5 #ASP,OD
    charge_score[6] = 1.0 #LYS,NZ
    charge_score[7] = -0.1 #PRO,N
    charge_score[8] = 0.0 #CA
    charge_score[9] = 0.0 #C
    charge_score[10] = -0.5 #O
    charge_score[11] = 0.5 #N
    
    return charge_score
end

function assign_Li_charge!(grid_real::AbstractArray{T}, 
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    charge_score::AbstractVector{T}) where {T}
    
    grid_real .= zero(T)

    spread_nearest_add!(grid_real, x, y, z, x_grid, y_grid, z_grid, charge_score)
    return nothing
end

function assign_Re_charge!(grid_real::AbstractArray{T}, 
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    charge_score::AbstractVector{T}) where {T}
    
    grid_real .= zero(T)

    spread_nearest_add!(grid_real, x, y, z, x_grid, y_grid, z_grid, charge_score)
    return nothing
end

# function spread_neighbors_add_potential!(grid::AbstractArray{T},
#     x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
#     x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
#     weight::AbstractVector{T}, rcut::AbstractVector{T}) where {T}

#     natom = length(x)
#     nx, ny, nz = size(grid)
#     for iatom = 1:natom
#         for ix = 1:nx
#             dx = x[iatom] - x_grid[ix]
#             if abs(dx) > rcut[iatom]
#                 continue
#             end
#             for iy = 1:ny
#                 dy = y[iatom] - y_grid[iy]
#                 if abs(dy) > rcut[iatom]
#                     continue
#                 end
#                 for iz = 1:nz
#                     dz = z[iatom] - z_grid[iz]
#                     if abs(dz) > rcut[iatom]
#                         continue
#                     end
#                     d = dx * dx + dy * dy + dz * dz
#                     if d < rcut[iatom] * rcut[iatom]
#                         grid[ix, iy, iz] += weight[iatom] / sqrt(d)
#                     end
#                 end
#             end
#         end
#     end

#     return nothing
# end

function calculate_distance!(grid::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    weight::AbstractVector{T}, rcut::AbstractVector{T}) where {T}
    
    grid .= zero(T)
    
    natom = length(x)
    nx, ny, nz = size(grid)
    for iatom = 1:natom
        for ix = 1:nx
            dx = x[iatom] - x_grid[ix]
            if abs(dx) > rcut[iatom]
                continue
            end
            for iy = 1:ny
                dy = y[iatom] - y_grid[iy]
                if abs(dy) > rcut[iatom]
                    continue
                end
                for iz = 1:nz
                    dz = z[iatom] - z_grid[iz]
                    if abs(dz) > rcut[iatom]
                        continue
                    end
                    d = dx * dx + dy * dy + dz * dz
                    if d < rcut[iatom] * rcut[iatom]
                        grid[ix, iy, iz] += sqrt(d)
                    end
                end
            end
        end
    end
    
    return nothing
end

function assign_Re_potential!(grid_imag::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    charge_score::AbstractVector{T}) where {T}
    
    grid_charge = similar(grid_imag)
    grid_dis = similar(grid_imag)
    
    
    grid_charge .= zero(T)
    grid_dis .= zero(T)
    
    rcut = similar(charge_score)
    rcut .= T(8.0)
    
    spread_nearest_add!(grid_charge, x, y, z, x_grid, y_grid, z_grid, charge_score)
    calculate_distance!(grid_dis, x, y, z, x_grid, y_grid, z_grid, charge_score, rcut)
    
    grid_imag = grid_charge ./ grid_dis
    
    return nothing
end

function assign_Li_potential!(grid_imag::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    charge_score::AbstractVector{T}) where {T}
    
    grid_charge = similar(grid_imag)
    grid_dis = similar(grid_imag)
    
    
    grid_charge .= zero(T)
    grid_dis .= zero(T)
    
    rcut = similar(charge_score)
    rcut .= T(8.0)
    
    spread_nearest_add!(grid_charge, x, y, z, x_grid, y_grid, z_grid, charge_score)
    calculate_distance!(grid_dis, x, y, z, x_grid, y_grid, z_grid, charge_score, rcut)
    
    grid_imag = grid_charge ./ grid_dis
    
    return nothing
end

# ============ cell 4 ============

################ grid

function generate_grid(receptor_org::TrjArray{T,U}, ligand_org::TrjArray{T,U}; iframe=1, spacing=1.2) where {T,U}
    receptor = deepcopy(receptor_org)
    ligand = deepcopy(ligand_org)
    decenter!(receptor)
    orient!(ligand)
    xmin_ligand = minimum(ligand.xyz[iframe, 1:3:end])
    xmax_ligand = maximum(ligand.xyz[iframe, 1:3:end])
    size_ligand = xmax_ligand - xmin_ligand

    xmin_receptor = minimum(receptor.xyz[iframe, 1:3:end])
    xmax_receptor = maximum(receptor.xyz[iframe, 1:3:end])
    xmin_grid = xmin_receptor - size_ligand - spacing
    xmax_grid = xmax_receptor + size_ligand + spacing

    ymin_receptor = minimum(receptor.xyz[iframe, 2:3:end])
    ymax_receptor = maximum(receptor.xyz[iframe, 2:3:end])
    ymin_grid = ymin_receptor - size_ligand - spacing
    ymax_grid = ymax_receptor + size_ligand + spacing

    zmin_receptor = minimum(receptor.xyz[iframe, 3:3:end])
    zmax_receptor = maximum(receptor.xyz[iframe, 3:3:end])
    zmin_grid = zmin_receptor - size_ligand - spacing
    zmax_grid = zmax_receptor + size_ligand + spacing

    x_grid = Array{T,1}(range(xmin_grid, xmax_grid, step=spacing))
    if typeof(receptor_org.xyz) <: CuArray
        x_grid = CuArray(x_grid)
    end

    y_grid = Array{T,1}(range(ymin_grid, ymax_grid, step=spacing))
    if typeof(receptor_org.xyz) <: CuArray
        y_grid = CuArray(y_grid)
    end

    z_grid = Array{T,1}(range(zmin_grid, zmax_grid, step=spacing))
    if typeof(receptor_org.xyz) <: CuArray
        z_grid = CuArray(z_grid)
    end

    nx = length(x_grid)
    ny = length(y_grid)
    nz = length(z_grid)

    grid_real = similar(receptor_org.xyz, (nx, ny, nz))
    grid_real .= zero(T)
    grid_imag = similar(receptor_org.xyz, (nx, ny, nz))
    grid_imag .= zero(T)

    return grid_real, grid_imag, x_grid, y_grid, z_grid
end

function spread_nearest_add!(grid::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    weight::AbstractVector{T}) where {T}

    natom = length(x)
    x_grid_delta = x_grid[2] - x_grid[1]
    y_grid_delta = y_grid[2] - y_grid[1]
    z_grid_delta = z_grid[2] - z_grid[1]

    x_grid_min = x_grid[1]
    y_grid_min = y_grid[1]
    z_grid_min = z_grid[1]

    for iatom = 1:natom
        ix = ceil(Int, (x[iatom] - x_grid_min) / x_grid_delta)
        iy = ceil(Int, (y[iatom] - y_grid_min) / y_grid_delta)
        iz = ceil(Int, (z[iatom] - z_grid_min) / z_grid_delta)
        grid[ix, iy, iz] += weight[iatom]
    end

    return nothing
end

function spread_nearest_substitute!(grid::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    weight::AbstractVector{T}) where {T}

    natom = length(x)
    x_grid_delta = x_grid[2] - x_grid[1]
    y_grid_delta = y_grid[2] - y_grid[1]
    z_grid_delta = z_grid[2] - z_grid[1]

    x_grid_min = x_grid[1]
    y_grid_min = y_grid[1]
    z_grid_min = z_grid[1]

    for iatom = 1:natom
        ix = ceil(Int, (x[iatom] - x_grid_min) / x_grid_delta)
        iy = ceil(Int, (y[iatom] - y_grid_min) / y_grid_delta)
        iz = ceil(Int, (z[iatom] - z_grid_min) / z_grid_delta)
        grid[ix, iy, iz] = weight[iatom]
    end

    return nothing
end

function spread_neighbors_add!(grid::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    weight::AbstractVector{T}, rcut::AbstractVector{T}) where {T}

    natom = length(x)
    nx, ny, nz = size(grid)
    for iatom = 1:natom
        for ix = 1:nx
            dx = x[iatom] - x_grid[ix]
            if abs(dx) > rcut[iatom]
                continue
            end
            for iy = 1:ny
                dy = y[iatom] - y_grid[iy]
                if abs(dy) > rcut[iatom]
                    continue
                end
                for iz = 1:nz
                    dz = z[iatom] - z_grid[iz]
                    if abs(dz) > rcut[iatom]
                        continue
                    end
                    d = dx * dx + dy * dy + dz * dz
                    if d < rcut[iatom] * rcut[iatom]
                        grid[ix, iy, iz] += weight[iatom]
                    end
                end
            end
        end
    end

    return nothing
end


function spread_neighbors_substitute!(grid::AbstractArray{T2},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    weight::AbstractVector{T}, rcut::AbstractVector{T}) where {T2,T}

    natom = length(x)
    nx, ny, nz = size(grid)

    for iatom = 1:natom
        for ix = 1:nx
            dx = x[iatom] - x_grid[ix]
            if abs(dx) > rcut[iatom]
                continue
            end
            for iy = 1:ny
                dy = y[iatom] - y_grid[iy]
                if abs(dy) > rcut[iatom]
                    continue
                end
                for iz = 1:nz
                    dz = z[iatom] - z_grid[iz]
                    if abs(dz) > rcut[iatom]
                        continue
                    end
                    d = dx * dx + dy * dy + dz * dz
                    if d < rcut[iatom] * rcut[iatom]
                        grid[ix, iy, iz] = weight[iatom]
                    end
                end
            end
        end
    end

    return nothing
end


#####################################################

function assign_sc_receptor_plus!(grid_real::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    radius::AbstractVector{T}, id_surface::AbstractVector) where {T}

    
    grid_real .= zero(T)

    
    x_s = x[id_surface]
    y_s = y[id_surface]
    z_s = z[id_surface]
    radius_s = radius[id_surface]
    weight_s = similar(radius_s)

    x_c = x[.!id_surface]
    y_c = y[.!id_surface]
    z_c = z[.!id_surface]
    radius_c = radius[.!id_surface]
    weight_c = similar(radius_c)

    weight_s .= T(1.0)
    spread_neighbors_substitute!(grid_real, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s .+ T(3.4))
    weight_s .= T(1.0)
    spread_neighbors_substitute!(grid_real, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s .* T(sqrt(0.8)))
    weight_c .= T(1.0)
    spread_neighbors_substitute!(grid_real, x_c, y_c, z_c, x_grid, y_grid, z_grid, weight_c, radius_c .* T(sqrt(1.5)))
     
    
    return nothing
end

function assign_sc_ligand_plus!(grid_real::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    radius::AbstractVector{T}, id_surface::AbstractVector) where {T}

    
    grid_real .= zero(T)

    
    x_s = x[id_surface]
    y_s = y[id_surface]
    z_s = z[id_surface]
    radius_s = radius[id_surface]
    weight_s = similar(radius_s)

    x_c = x[.!id_surface]
    y_c = y[.!id_surface]
    z_c = z[.!id_surface]
    radius_c = radius[.!id_surface]
    weight_c = similar(radius_c)

    weight_s .= T(1.0)
    spread_neighbors_substitute!(grid_real, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s)

    weight_c .= T(1.0)
    spread_neighbors_substitute!(grid_real, x_c, y_c, z_c, x_grid, y_grid, z_grid, weight_c, radius_c .* T(sqrt(1.5)))
     
    
    return nothing
end

function assign_sc_receptor_minus!(grid_imag::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    radius::AbstractVector{T}, id_surface::AbstractVector) where {T}

    
    grid_imag .= zero(T)

    
    x_s = x[id_surface]
    y_s = y[id_surface]
    z_s = z[id_surface]
    radius_s = radius[id_surface]
    weight_s = similar(radius_s)

    x_c = x[.!id_surface]
    y_c = y[.!id_surface]
    z_c = z[.!id_surface]
    radius_c = radius[.!id_surface]
    weight_c = similar(radius_c)

    weight_s .= T(3.5)
    spread_neighbors_substitute!(grid_imag, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s .+ T(3.4))
    weight_s .= T(12.25)
    spread_neighbors_substitute!(grid_imag, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s .* T(sqrt(0.8)))
    weight_c .= T(12.25)
    spread_neighbors_substitute!(grid_imag, x_c, y_c, z_c, x_grid, y_grid, z_grid, weight_c, radius_c .* T(sqrt(1.5)))
     
    
    return nothing
end
    
function assign_sc_ligand_minus!(grid_imag::AbstractArray{T},
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    radius::AbstractVector{T}, id_surface::AbstractVector) where {T}

    
    grid_imag .= zero(T)

    
    x_s = x[id_surface]
    y_s = y[id_surface]
    z_s = z[id_surface]
    radius_s = radius[id_surface]
    weight_s = similar(radius_s)

    x_c = x[.!id_surface]
    y_c = y[.!id_surface]
    z_c = z[.!id_surface]
    radius_c = radius[.!id_surface]
    weight_c = similar(radius_c)

    weight_s .= T(3.5)
    spread_neighbors_substitute!(grid_imag, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s)

    weight_c .= T(12.25)
    spread_neighbors_substitute!(grid_imag, x_c, y_c, z_c, x_grid, y_grid, z_grid, weight_c, radius_c .* T(sqrt(1.5)))
     
    
    return nothing
end
###################################################

# function assign_sc_receptor!(grid_real::AbstractArray{T}, grid_imag::AbstractArray{T},
#     x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
#     x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
#     radius::AbstractVector{T}, id_surface::AbstractVector) where {T}

#     grid_real .= zero(T)
#     grid_imag .= zero(T)

#     x_s = x[id_surface]
#     y_s = y[id_surface]
#     z_s = z[id_surface]
#     radius_s = radius[id_surface]
#     weight_s = similar(radius_s)

#     x_c = x[.!id_surface]
#     y_c = y[.!id_surface]
#     z_c = z[.!id_surface]
#     radius_c = radius[.!id_surface]
#     weight_c = similar(radius_c)

#     weight_s .= T(1.0)
#     spread_neighbors_substitute!(grid_real, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s .+ T(3.4))
#     weight_s .= T(0.0)
#     spread_neighbors_substitute!(grid_real, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s .* T(sqrt(0.8)))
#     weight_c .= T(0.0)
#     spread_neighbors_substitute!(grid_real, x_c, y_c, z_c, x_grid, y_grid, z_grid, weight_c, radius_c .* T(sqrt(1.5)))

#     weight_s .= T(9.0)
#     spread_neighbors_substitute!(grid_imag, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s .* T(sqrt(0.8)))
#     weight_c .= T(9.0)
#     spread_neighbors_substitute!(grid_imag, x_c, y_c, z_c, x_grid, y_grid, z_grid, weight_c, radius_c .* T(sqrt(1.5)))
     
#     return nothing
# end

# function assign_sc_ligand!(grid_real::AbstractArray{T}, grid_imag::AbstractArray{T},
#     x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
#     x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
#     radius::AbstractVector{T}, id_surface::AbstractVector) where {T}

#     grid_real .= zero(T)
#     grid_imag .= zero(T)

#     x_s = x[id_surface]
#     y_s = y[id_surface]
#     z_s = z[id_surface]
#     radius_s = radius[id_surface]
#     weight_s = similar(radius_s)

#     x_c = x[.!id_surface]
#     y_c = y[.!id_surface]
#     z_c = z[.!id_surface]
#     radius_c = radius[.!id_surface]
#     weight_c = similar(radius_c)
    
#     weight_s .= T(1.0)
#     spread_neighbors_substitute!(grid_real, x_s, y_s, z_s, x_grid, y_grid, z_grid, weight_s, radius_s)

#     weight_c .= T(9.0)
#     spread_neighbors_substitute!(grid_imag, x_c, y_c, z_c, x_grid, y_grid, z_grid, weight_c, radius_c .* T(sqrt(1.5)))

#     # check_neighbors_ligand!(grid_real, grid_imag)

#     return nothing
# end

function assign_Rij!(grid_real::AbstractArray{T}, 
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
    iface_ij::AbstractVector{T}) where {T}

    grid_real .= zero(T)

    radius = similar(iface_ij)
    radius .= T(6.0)
    spread_neighbors_add!(grid_real, x, y, z, x_grid, y_grid, z_grid, iface_ij, radius)

    return nothing
end

function assign_Li!(grid_real::AbstractArray{T}, 
    x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
    x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T}
    ) where {T}

    grid_real .= zero(T)

    radius = similar(x)
    radius .= T(1.0)
    spread_nearest_substitute!(grid_real, x, y, z, x_grid, y_grid, z_grid, radius)

    return nothing
end

# function assign_ds!(grid_real::AbstractArray{T}, grid_imag::AbstractArray{T},
#     x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T},
#     x_grid::AbstractVector{T}, y_grid::AbstractVector{T}, z_grid::AbstractVector{T},
#     iface_ij::AbstractVector{T}) where {T}

#     grid_real .= zero(T)
#     grid_imag .= zero(T)

#     radius = similar(iface_ij)
#     radius .= T(6.0)
#     spread_neighbors_add!(grid_real, x, y, z, x_grid, y_grid, z_grid, iface_ij, radius)

#     radius .= T(1.0)
#     spread_nearest_substitute!(grid_imag, x, y, z, x_grid, y_grid, z_grid, radius)

#     return nothing
# end

################ docking

function docking_score_elec(receptor_org::TrjArray{T,U},
        ligands_org::TrjArray{T,U}, alpha::T, iface_ij::AbstractArray{T}, 
        beta::T, charge_score::AbstractArray{T}) where {T,U}
    spacing = 3.0
    receptor = deepcopy(receptor_org)
    ligands = deepcopy(ligands_org)

    decenter!(receptor)
    decenter!(ligands)

    grid_real, grid_imag, x_grid, y_grid, z_grid = generate_grid(receptor, ligands, spacing=spacing)
    nxyz = T(prod(size(grid_real)))
    
    grid_iface = similar(grid_real)
    grid_elec = similar(grid_imag)

    com = centerofmass(receptor)
    receptor.xyz[:, 1:3:end] .= receptor.xyz[:, 1:3:end] .- com.xyz[:, 1:1]
    receptor.xyz[:, 2:3:end] .= receptor.xyz[:, 2:3:end] .- com.xyz[:, 2:2]
    receptor.xyz[:, 3:3:end] .= receptor.xyz[:, 3:3:end] .- com.xyz[:, 3:3]
    ligands.xyz[:, 1:3:end] .= ligands.xyz[:, 1:3:end] .- com.xyz[:, 1:1]
    ligands.xyz[:, 2:3:end] .= ligands.xyz[:, 2:3:end] .- com.xyz[:, 2:2]
    ligands.xyz[:, 3:3:end] .= ligands.xyz[:, 3:3:end] .- com.xyz[:, 3:3]

    # receptor.mass .= iface_ij[receptor.atomtype_id]
    # ligands.mass .= iface_ij[ligands.atomtype_id]

    x = receptor.xyz[1, 1:3:end]
    y = receptor.xyz[1, 2:3:end]
    z = receptor.xyz[1, 3:3:end]
    id_surface = receptor.sasa .> 1.0
    
#     print(typeof(receptor.xyz),size(receptor.xyz))
#     print(typeof(ligands.xyz),size(ligands.xyz))    
#     print_all!(x, y, z, x_grid, y_grid, z_grid)   
    
     
#     function calculate_atom_distances(receptor::TrjArray{T,U}, ligands::TrjArray{T,U}, 
#             x::AbstractVector{T}, y::AbstractVector{T}, z::AbstractVector{T})
#         m, n = length(x), length(y)
#         distances = similar(x, m, n)
#         for i in 1:m
#             for j in 1:n
#                 distances[i, j] = sqrt((receptor.xyz[i, 1] - ligands.xyz[j, 1])^2 +
#                            (receptor.xyz[i, 2] - ligands.xyz[j, 2])^2 +
#                                   (receptor.xyz[i, 3] - ligands.xyz[j, 3])^2)
#             end
#         end
 
#         return distances
#     end
    
#     atom_distances = calculate_atom_distances(receptor, ligands, x, y, z)
    

    ##########################################
    
    assign_sc_receptor_plus!(grid_real, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
    assign_sc_receptor_minus!(grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
    #assign_sc_receptor!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
    grid_sc_receptor = grid_real .+ im .* grid_imag
    
    ##########################################

    # assign_ds!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.mass)
    # grid_ds_receptor = grid_real .+ im .* grid_imag
    
#     print_all!(x, y, z, x_grid, y_grid, z_grid)   

    x = ligands.xyz[1, 1:3:end]
    y = ligands.xyz[1, 2:3:end]
    z = ligands.xyz[1, 3:3:end]

    x2 = receptor.xyz[1, 1:3:end]
    y2 = receptor.xyz[1, 2:3:end]
    z2 = receptor.xyz[1, 3:3:end]

    id_surface = ligands.sasa .> 1.0

    grid_sc_ligand = deepcopy(grid_sc_receptor)
    # grid_iface_ligand = deepcopy(grid_real)
    # grid_iface_receptor = deepcopy(grid_real)
    score_sc = similar(grid_real, ligands.nframe)
    score_iface = similar(grid_real, ligands.nframe)
    score_iface .= zero(T)
    score_elec = similar(grid_real, ligands.nframe)
    score_elec .= zero(T)
    score_total = similar(grid_real, ligands.nframe)

    @showprogress for iframe = 1:ligands.nframe
        x .= ligands.xyz[iframe, 1:3:end]
        y .= ligands.xyz[iframe, 2:3:end]
        z .= ligands.xyz[iframe, 3:3:end]

        ###################################################
        
#         assign_sc_ligand!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
#         grid_sc_ligand .= grid_real .+ im .* grid_imag
#         multi = grid_sc_receptor .* grid_sc_ligand
#         score_sc[iframe] = sum(real.(multi)) - sum(imag.(multi))

        assign_sc_ligand_plus!(grid_real, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
        assign_sc_ligand_minus!(grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
        grid_sc_ligand .= grid_real .+ im .* grid_imag 
        multi = grid_sc_receptor .* grid_sc_ligand
        score_sc[iframe] = sum(real.(multi)) - sum(imag.(multi)) 
        
        
        ##################################################        
        
        # assign_ds!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.mass)
        # grid_ds_ligand .= grid_real .+ im .* grid_imag
        # multi = grid_ds_receptor .* grid_ds_ligand
        # score_ds[iframe] = T(0.5) * sum(imag(multi))

        for i = 1:12
            idx = ligands.atomtype_id .== i 
            assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
            grid_iface = grid_real
            for j = 1:12
                k = 12 * (j-1) + i 
                receptor.mass .= iface_ij[k]
                idx = receptor.atomtype_id .== j
                assign_Rij!(grid_imag, x2[idx], y2[idx], z2[idx], x_grid, y_grid, z_grid, receptor.mass[idx])
                score_iface[iframe] += sum(grid_real .* grid_imag)
            end
        end 
        
        for l = 1:11
            idx = ligands.atomtype_id .== l
            idx2 = receptor.atomtype_id .== l
            ligands.mass .= charge_score[l]
            receptor.mass .= charge_score[l]
            assign_Li_charge!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid, ligands.mass[idx])
            assign_Re_potential!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])
            score_elec[iframe] += sum(grid_real .* grid_imag)
            grid_elec = grid_imag            
        end

        score_total[iframe] = alpha .* score_sc[iframe] .+ score_iface[iframe] .+ beta .* score_elec[iframe]
    end

#     print_all!(x, y, z, x_grid, y_grid, z_grid)
    
    return score_total
end


# ============ cell 5 ============
function docking_score(receptor_org::TrjArray{T,U},
        ligands_org::TrjArray{T,U}, alpha::T, iface_ij::AbstractArray{T}
        ) where {T,U}
    spacing = 3.0
    receptor = deepcopy(receptor_org)
    ligands = deepcopy(ligands_org)

    decenter!(receptor)
    decenter!(ligands)

    grid_real, grid_imag, x_grid, y_grid, z_grid = generate_grid(receptor, ligands, spacing=spacing)
    nxyz = T(prod(size(grid_real)))

    
    com = centerofmass(receptor)
    receptor.xyz[:, 1:3:end] .= receptor.xyz[:, 1:3:end] .- com.xyz[:, 1:1]
    receptor.xyz[:, 2:3:end] .= receptor.xyz[:, 2:3:end] .- com.xyz[:, 2:2]
    receptor.xyz[:, 3:3:end] .= receptor.xyz[:, 3:3:end] .- com.xyz[:, 3:3]
    ligands.xyz[:, 1:3:end] .= ligands.xyz[:, 1:3:end] .- com.xyz[:, 1:1]
    ligands.xyz[:, 2:3:end] .= ligands.xyz[:, 2:3:end] .- com.xyz[:, 2:2]
    ligands.xyz[:, 3:3:end] .= ligands.xyz[:, 3:3:end] .- com.xyz[:, 3:3]

    # receptor.mass .= iface_ij[receptor.atomtype_id]
    # ligands.mass .= iface_ij[ligands.atomtype_id]

    x = receptor.xyz[1, 1:3:end]
    y = receptor.xyz[1, 2:3:end]
    z = receptor.xyz[1, 3:3:end]
    id_surface = receptor.sasa .> 1.0

    ##########################################
    
    assign_sc_receptor_plus!(grid_real, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
    assign_sc_receptor_minus!(grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
    #assign_sc_receptor!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
    grid_sc_receptor = grid_real .+ im .* grid_imag
    
    ##########################################

    # assign_ds!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.mass)
    # grid_ds_receptor = grid_real .+ im .* grid_imag

    x = ligands.xyz[1, 1:3:end]
    y = ligands.xyz[1, 2:3:end]
    z = ligands.xyz[1, 3:3:end]

    x2 = receptor.xyz[1, 1:3:end]
    y2 = receptor.xyz[1, 2:3:end]
    z2 = receptor.xyz[1, 3:3:end]

    id_surface = ligands.sasa .> 1.0

    grid_sc_ligand = deepcopy(grid_sc_receptor)
    # grid_iface_ligand = deepcopy(grid_real)
    # grid_iface_receptor = deepcopy(grid_real)
    score_sc = similar(grid_real, ligands.nframe)
    score_iface = similar(grid_real, ligands.nframe)
    score_iface .= zero(T)
    score_total = similar(grid_real, ligands.nframe)

    @showprogress for iframe = 1:ligands.nframe
        x .= ligands.xyz[iframe, 1:3:end]
        y .= ligands.xyz[iframe, 2:3:end]
        z .= ligands.xyz[iframe, 3:3:end]

        ###################################################
        
#         assign_sc_ligand!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
#         grid_sc_ligand .= grid_real .+ im .* grid_imag
#         multi = grid_sc_receptor .* grid_sc_ligand
#         score_sc[iframe] = sum(real.(multi)) - sum(imag.(multi))

        assign_sc_ligand_plus!(grid_real, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
        assign_sc_ligand_minus!(grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
        grid_sc_ligand .= grid_real .+ im .* grid_imag 
        multi = grid_sc_receptor .* grid_sc_ligand
        score_sc[iframe] = sum(real.(multi)) - sum(imag.(multi)) 
        
        
        ##################################################        
        
        # assign_ds!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.mass)
        # grid_ds_ligand .= grid_real .+ im .* grid_imag
        # multi = grid_ds_receptor .* grid_ds_ligand
        # score_ds[iframe] = T(0.5) * sum(imag(multi))

        for i = 1:12
            idx = ligands.atomtype_id .== i 
            assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
            for j = 1:12
                k = 12 * (j-1) + i 
                receptor.mass .= iface_ij[k]
                idx = receptor.atomtype_id .== j
                assign_Rij!(grid_imag, x2[idx], y2[idx], z2[idx], x_grid, y_grid, z_grid, receptor.mass[idx])
                score_iface[iframe] += sum(grid_real .* grid_imag)
            end
        end 


        score_total[iframe] = alpha .* score_sc[iframe] .+ score_iface[iframe] 
    end

    return score_total
end


# ============ cell 6 ============
function ChainRulesCore.rrule(::typeof(docking_score_elec), receptor_org::TrjArray{T,U}, ligands_org::TrjArray{T,U}, alpha::T, iface_ij::AbstractVector{T}, beta::T, charge_score::AbstractArray{T}) where {T,U} 
    spacing = 1.5 
    receptor = deepcopy(receptor_org) 
    ligands = deepcopy(ligands_org)
   
    decenter!(receptor)
    decenter!(ligands)

    grid_real, grid_imag, x_grid, y_grid, z_grid = MDToolbox.generate_grid(receptor, ligands, spacing=spacing)
    nxyz = T(prod(size(grid_real)))

    com = centerofmass(receptor)
    receptor.xyz[:, 1:3:end] .= receptor.xyz[:, 1:3:end] .- com.xyz[:, 1:1]
   receptor.xyz[:, 2:3:end] .= receptor.xyz[:, 2:3:end] .- com.xyz[:, 2:2]
   receptor.xyz[:, 3:3:end] .= receptor.xyz[:, 3:3:end] .- com.xyz[:, 3:3]
   ligands.xyz[:, 1:3:end] .= ligands.xyz[:, 1:3:end] .- com.xyz[:, 1:1]
   ligands.xyz[:, 2:3:end] .= ligands.xyz[:, 2:3:end] .- com.xyz[:, 2:2]
   ligands.xyz[:, 3:3:end] .= ligands.xyz[:, 3:3:end] .- com.xyz[:, 3:3]

#   receptor.mass .= iface_ij[receptor.atomtype_id]
#   ligands.mass .= iface_ij[ligands.atomtype_id]

   x = receptor.xyz[1, 1:3:end]
   y = receptor.xyz[1, 2:3:end]
   z = receptor.xyz[1, 3:3:end]
   id_surface = receptor.sasa .> 1.0

   assign_sc_receptor_plus!(grid_real, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
   assign_sc_receptor_minus!(grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
#   MDToolbox.assign_sc_receptor!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.radius, id_surface)
   grid_sc_receptor = grid_real .+ im .* grid_imag

#   assign_ds!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, receptor.mass)
#   grid_ds_receptor = grid_real .+ im .* grid_imag

   x = ligands.xyz[1, 1:3:end]
   y = ligands.xyz[1, 2:3:end]
   z = ligands.xyz[1, 3:3:end]

   x2 = receptor.xyz[1, 1:3:end]
   y2 = receptor.xyz[1, 2:3:end]
   z2 = receptor.xyz[1, 3:3:end]

   id_surface = ligands.sasa .> 1.0

   grid_sc_ligand = deepcopy(grid_sc_receptor)
   
    score_sc = similar(grid_real, ligands.nframe)
    score_sc .= zero(T)
    score_iface = similar(grid_real, ligands.nframe)
    score_iface .= zero(T)
    score_elec = similar(grid_real, ligands.nframe)
    score_elec .= zero(T)
    data1 = similar(grid_real, length(iface_ij), ligands.nframe)
    data1 .= zero(T)
    data2 = similar(grid_real, length(charge_score), ligands.nframe)
    data2 .= zero(T)
    data3 = similar(grid_real, length(charge_score), ligands.nframe)
    data3 .= zero(T)
    score_total = similar(grid_real, ligands.nframe)
    cb = similar(grid_real, length(charge_score), ligands.nframe)

    score_for_ifacescore = similar(grid_real, length(iface_ij), ligands.nframe)
    score_for_chargescore = similar(grid_real, length(charge_score), ligands.nframe)
#     score_for_ifacescore = zeros(T, 12 * 12, ligands.nframe)
#     score_for_chargescore = zeros(T, 11, ligands.nframe)
    print("score")
   @showprogress for iframe = 1:ligands.nframe
        x .= ligands.xyz[iframe, 1:3:end]
        y .= ligands.xyz[iframe, 2:3:end]
        z .= ligands.xyz[iframe, 3:3:end]
        
        assign_sc_ligand_plus!(grid_real, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
        assign_sc_ligand_minus!(grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
#     MDToolbox.assign_sc_ligand!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.radius, id_surface)
        grid_sc_ligand .= grid_real .+ im .* grid_imag
        multi = grid_sc_receptor .* grid_sc_ligand
        score_sc[iframe] = sum(real.(multi)) - sum(imag.(multi))       

       # assign_ds!(grid_real, grid_imag, x, y, z, x_grid, y_grid, z_grid, ligands.mass)
       # grid_ds_ligand .= grid_real .+ im .* grid_imag
       # multi = grid_ds_receptor .* grid_ds_ligand
       # score_ds[iframe] = T(0.5) * sum(imag(multi))
  
        for i = 1:12
            idx = ligands.atomtype_id .== i
            if any(idx)
                assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
#          MDToolbox.assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
                for j = 1:12
                    k = 12 * (j-1) + i                   
                    receptor.mass .= iface_ij[k]
                    idx2 = receptor.atomtype_id .== j
                    if any(idx)
                        assign_Rij!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])
#                MDToolbox.assign_Rij!(grid_imag, x2[idx], y2[idx], z2[idx], x_grid, y_grid, z_grid, receptor.mass[idx])
                        score_iface[iframe] += sum(grid_real .* grid_imag) 
                    end 
                end
            end
        end  
        for l = 1:11
            idx = ligands.atomtype_id .== l
            idx2 = receptor.atomtype_id .== l
            if any(idx)
                if any(idx2)
                    ligands.mass .= charge_score[l]
                    receptor.mass .= charge_score[l]
                    assign_Li_charge!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid, ligands.mass[idx])
                    assign_Re_potential!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])                    
                    score_elec[iframe] += sum(grid_real .* grid_imag)
#                     grid_imag .= zero(T)
#                     calculate_distance!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2]) 
#                     dis[iframe] += sum(grid_imag)    
                end  
            end
        end   
        score_total[iframe] = alpha .* score_sc[iframe] .+ score_iface[iframe] .+ beta .* score_elec[iframe]
  end 
  score_sc_old = deepcopy(score_sc)
  score_elec_old = deepcopy(score_elec)
    
    
    ####################################################
    print("score_for_iface")
    @showprogress for iframe = 1:ligands.nframe              
        receptor.mass .= zero(T)
        ligands.mass .= zero(T)
        x .= ligands.xyz[iframe, 1:3:end]
        y .= ligands.xyz[iframe, 2:3:end]
        z .= ligands.xyz[iframe, 3:3:end]
    
        for i = 1:12
            idx = ligands.atomtype_id .== i
            ligands.mass[idx] .= one(T)
            if any(idx)
                assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
#          MDToolbox.assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
                for j = 1:12
                    k = 12 * (j-1) + i  
                    k_dual = 12 * (i-1) + j
                    idx2 = receptor.atomtype_id .== j
                    receptor.mass[idx2] .= one(T)
                    if any(idx2)
                        assign_Rij!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])
#                MDToolbox.assign_Rij!(grid_imag, x2[idx], y2[idx], z2[idx], x_grid, y_grid, z_grid, receptor.mass[idx])
                        tmp = sum(grid_real .* grid_imag)
                        data1[k, iframe] = tmp
                        data1[k_dual, iframe] = tmp 
                        score_for_ifacescore = data1
                    end 
                end
            end
        end  
    end

    
###########################################################
#      print("score_for_iface")
#     @showprogress for itype = 1:length(iface_ij)
#         receptor.mass .= zero(T)
#         ligands.mass .= zero(T)
#         idx = receptor.atomtype_id .== iface_ij[itype]
#         receptor.mass[idx] .= one(T)

#         x = receptor.xyz[1, 1:3:end]
#         y = receptor.xyz[1, 2:3:end]
#         z = receptor.xyz[1, 3:3:end]
#         id_surface = receptor.sasa .> 1.0
    
#         assign_Rij!(grid_imag, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid, receptor.mass[idx])

#         idx = ligands.atomtype_id .== iface_ij[itype]
#         ligands.mass[idx] .= one(T)

#         x = ligands.xyz[1, 1:3:end]
#         y = ligands.xyz[1, 2:3:end]
#         z = ligands.xyz[1, 3:3:end]
#         id_surface = ligands.sasa .> 1.0

#         for iframe = 1:ligands.nframe
#             x .= ligands.xyz[iframe, 1:3:end]
#             y .= ligands.xyz[iframe, 2:3:end]
#             z .= ligands.xyz[iframe, 3:3:end]

#             for i = 1:12
#                 idx = ligands.atomtype_id .== i
#                 if any(idx)
#                     assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
# #          MDToolbox.assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
#                     for j = 1:12
#                         k = 12 * (j-1) + i  
#                         k_dual = 12 * (i-1) + j
#                         idx2 = receptor.atomtype_id .== j
#                         if any(idx2)
#                             assign_Rij!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])
# #                MDToolbox.assign_Rij!(grid_imag, x2[idx], y2[idx], z2[idx], x_grid, y_grid, z_grid, receptor.mass[idx])
#                             tmp = sum(grid_real .* grid_imag)
#                             data1[k, iframe] = tmp
#                             data1[k_dual, iframe] = tmp 
#                         end 
#                     end
#                 end
#             end 
#             score_for_ifacescore[itype, iframe] = data1[iframe] 
#        end
#     end

###########################################################score_for_ifacescore
    print("score_for_chargescore")
    @showprogress for iframe = 1:ligands.nframe
        receptor.mass .= zero(T)
        ligands.mass .= zero(T)
        x .= ligands.xyz[iframe, 1:3:end]
        y .= ligands.xyz[iframe, 2:3:end]
        z .= ligands.xyz[iframe, 3:3:end]
    
        for l = 1:11
            idx = ligands.atomtype_id .== l
            idx2 = receptor.atomtype_id .== l
            if any(idx)
                if any(idx2)
                    ligands.mass[idx] .= charge_score[l]
                    receptor.mass[idx2] .= one(T)
                    assign_Re_charge!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid, ligands.mass[idx])
                    assign_Li_potential!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])                    
                    data2[l, iframe] = sum(grid_real .* grid_imag)
                    score_for_chargescore = data2
#                     grid_imag .= zero(T)
#                     calculate_distance!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2]) 
#                     dis[iframe] += sum(grid_imag) 
                end 
            end
        end  
            # score_for_acescore[iscore, iframe] = alpha .* score_sc[iframe] .+ score_ds[iframe] 
    end
    @showprogress for iframe = 1:ligands.nframe       
        receptor.mass .= zero(T)
        ligands.mass .= zero(T)
        x .= ligands.xyz[iframe, 1:3:end]
        y .= ligands.xyz[iframe, 2:3:end]
        z .= ligands.xyz[iframe, 3:3:end]
    
        for l = 1:11
            idx = ligands.atomtype_id .== l
            idx2 = receptor.atomtype_id .== l
            if any(idx)
                if any(idx2)
                    ligands.mass[idx] .= one(T)
                    receptor.mass[idx2] .= charge_score[l]
                    assign_Li_charge!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid, ligands.mass[idx])
                    assign_Re_potential!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])                    
                    data3[l, iframe] = sum(grid_real .* grid_imag)
                    score_for_chargescore += data3
#                     grid_imag .= zero(T)
#                     calculate_distance!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2]) 
#                     dis[iframe] += sum(grid_imag) 
                end 
            end
        end  
            # score_for_acescore[iscore, iframe] = alpha .* score_sc[iframe] .+ score_ds[iframe] 
    end
    
    
#     score_for_ifacescore = iface_ij
#     score_for_chargescore = charge_score
    
#     score_for_ifacescore = reshape(repeat(iface_ij, 100), 144, 100)
#     score_for_chargescore = reshape(repeat(charge_score, 100), 11, 100)    

#     println("beta=",beta)
#     println("bata=",typeof(beta))
#     println("beta=",size(beta))
#     println("iface=",score_for_ifacescore)
#     println("iface=",typeof(score_for_ifacescore))
#     println("iface=",size(score_for_ifacescore))     
#     println("charge=",score_for_chargescore)
#     println("charge=",typeof(score_for_chargescore))
#     println("charge=",size(score_for_chargescore)) 
    
  
#     for iframe = 1:ligands.nframe
#     x .= ligands.xyz[iframe, 1:3:end]
#     y .= ligands.xyz[iframe, 2:3:end]
#     z .= ligands.xyz[iframe, 3:3:end]
#     for i = 1:12
#       idx = ligands.atomtype_id .== i
#       if any(idx)
#         assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
# #        MDToolbox.assign_Li!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid)
#         for j = 1:12
#            k = 12 * (j-1) + i
#           receptor.mass .= iface_ij[k]
#           idx = receptor.atomtype_id .== j
#           if any(idx)
#           assign_Rij!(grid_imag, x2[idx], y2[idx], z2[idx], x_grid, y_grid, z_grid, receptor.mass[idx])
# #          MDToolbox.assign_Rij!(grid_imag, x2[idx], y2[idx], z2[idx], x_grid, y_grid, z_grid, receptor.mass[idx])
#           score_iface[iframe] += sum(grid_real .* grid_imag)
#         end
#       end
#     end
#     for l = 1:11
#       idx = ligands.atomtype_id .== l
#       idx2 = receptor.atomtype_id .== l
#       ligands.mass .= charge_score[l]
#       receptor.mass .= charge_score[l]
#       assign_Li_charge!(grid_real, x[idx], y[idx], z[idx], x_grid, y_grid, z_grid, ligands.mass[idx])
#       assign_Re_potential!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])
#       score_elec[iframe] += sum(grid_real .* grid_imag)
#     end
#   end
   
    
            
#     for l = 1:11
#         idx = ligands.atomtype_id .== l
#         idx2 = receptor.atomtype_id .== l
#         if any(idx)
#             if any(idx2)
#                 ligands.mass .= charge_score[l]
#                 receptor.mass .= charge_score[l]
#                 calculate_distance!(grid_imag, x2[idx2], y2[idx2], z2[idx2], x_grid, y_grid, z_grid, receptor.mass[idx2])
#                 d = sum(grid_imag)
#             end
#         end  
#     end 
 

  function pullback(ybar)  
        println("loss")   
        sber = NoTangent()
        rber = NoTangent()
        lber = NoTangent()
        aber = sum(score_sc_old .* ybar)
        ifber = zeros(144)
        for a in 1:144
            sfi = score_for_ifacescore[a,:]
#             println(sfi)
            ifber[a] = LinearAlgebra.dot(sfi,ybar)
        end
        bber = sum(score_elec_old .* ybar)
        chber = zeros(11)
        for b in 1:11 
            sfc = score_for_chargescore[b,:]
            cb[b] = LinearAlgebra.dot(sfc,ybar)
            chber[b] = beta .* cb[b]
        end
#        println(aber)
#       println(typeof(aber))
#       println(size(aber))     
#         println(score_for_ifacescore)
#       println(typeof(score_for_ifacescore))
#       println(size(score_for_ifacescore))
#       println(ifber)
#       println(typeof(ifber))
#       println(size(ifber))
#       println(bber)
#       println(typeof(bber))
#       println(size(bber))
      println(chber)
      println(typeof(chber))
      println(size(chber))
            
      return sber, rber, lber, aber, ifber, bber, chber
  end

  return score_total, pullback
end
