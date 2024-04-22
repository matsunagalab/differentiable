"""
    tram(c_ijk, N_ik, b_kln, index_of_cluster_kn; tol=1e-10, max_iter=20)

Estimates the free energy differences by transition-based reweighting analysis method (TRAM).

## Arguments
- `c_ijk::AbstractArray`: Array containing the counts of transitions from state `i` to state `j` in ensemble `k`.
- `N_ik::AbstractArray`: Array containing the counts of samples in state `i` in ensemble `k`.
- `b_kln::AbstractArray`: Array of bias factors of state `k` evaluated at state `l`.
- `index_of_cluster_kn::AbstractArray`: Array fo index of the state to which the nth sample of ensemble `k` belongs.. 

## Optional Arguments
- `tol::Real = 1e-10`: Tolerance value for convergence criteria.
- `max_iter::Integer = 20`: Maximum number of iterations allowed.
"""
function tram(c_ijk, N_ik, b_kln, index_of_cluster_kn; tol=1e-10, max_iter=20)
    f_ik, v_ik, R_ik = tram_init(c_ijk, N_ik)

    for i in 1:max_iter
        R_ik = tram_R_ik(c_ijk, v_ik, f_ik, N_ik)
        for j in 1:100 #この100はなんとなくの値, f_ikの更新に比べ計算量が少ないため適当に大きい値を設定している
            v_ik = tram_v_new_ik(c_ijk, v_ik, f_ik)
        end
        f_new_ik = tram_f_new_ik(f_ik, R_ik, b_kln, N_ik, index_of_cluster_kn)

        max_dif = maximum(f_ik .- f_new_ik)
        println("iteration = $(i), delta = $(max_dif)")
        if(max_dif < tol)
            f_ik = f_new_ik
            break
        end
        f_ik = f_new_ik
    end
    
    for j in 1:100 # v_ikも一応更新する
        v_ik = tram_v_new_ik(c_ijk, v_ik, f_ik)
    end

    return f_ik, v_ik, R_ik
end

function tram_p_ijk(c_ijk, v_ik, f_ik)
    m, K = size(f_ik)

    p_ijk = zeros(Float64, m, m, K)
    for k in 1:K
        for i in 1:m
            for j in 1:m
                p_ijk[i, j, k] = (c_ijk[i, j, k] + c_ijk[j, i, k]) / (exp(f_ik[j, k] - f_ik[i, k]) * v_ik[j, k] + v_ik[i, k])
            end
        end
    end
    return p_ijk
end

function tram_R_ik(c_ijk, v_ik, f_ik, N_ik)
    m, K = size(N_ik)

    R_ik = zeros(Float64, m, K)
    for k in 1:K
        for i in 1:m   
            for j in 1:m
                R_ik[i, k] += (c_ijk[i, j, k] + c_ijk[j, i, k]) * v_ik[j, k] / (v_ik[j, k] + exp(f_ik[i, k] - f_ik[j, k]) * v_ik[i, k])
            end
            R_ik[i, k] += N_ik[i, k]
            for j in 1:m
                R_ik[i, k] -= c_ijk[j, i, k]
            end
        end
    end

    return R_ik
end

function tram_v_new_ik(c_ijk, v_ik, f_ik)
    m, K = size(f_ik)

    v_new_ik = zeros(Float64, m, K)
    for k in 1:K
        for i in 1:m
            for j in 1:m
                v_new_ik[i, k] += v_ik[i, k] * (c_ijk[i, j, k] + c_ijk[j, i, k]) / (exp(f_ik[j, k] - f_ik[i, k]) * v_ik[j, k] + v_ik[i, k])
            end
        end
    end
    return v_new_ik
end

# f^k,newの計算では、Σ_(x∈X_i)があるため、index_of_cluster_knを用いる
# b_klnはkアンサンブルをl番目のポテンシャルで評価したもの

function tram_f_new_ik(f_ik, R_ik, b_kln, N_ik, index_of_cluster_kn)
    m, K = size(N_ik)
    f_new_ik = zeros(Float64, m, K)

    log_w_ikn = tram_log_w_ikn(f_ik, R_ik, b_kln, N_ik, index_of_cluster_kn)
    for k in 1:K
        for i in 1:m
            tmp = Array{Float64}(undef, 0)
            for n in 1:length(index_of_cluster_kn[k, :])
                if(index_of_cluster_kn[k, n] == i)
                    append!(tmp, log_w_ikn[i, k, n])
                end
            end
            f_new_ik[i, k] = - MDToolbox.logsumexp(tmp)
        end
    end
    
    f_new_ik = normalize_f_ik(f_new_ik)
    return f_new_ik
end

function tram_log_w_ikn(f_ik, R_ik, b_kln, N_ik, index_of_cluster_kn)
    m, K = size(R_ik)
    log_w_ikn = zeros(Float64, m, K, size(b_kln, 3))

    for k in 1:K
        for i in 1:m
            for n in 1:size(b_kln, 3)
                if(index_of_cluster_kn[k, n] == i)
                    x = log.(R_ik[i, :]) + f_ik[i, :] - (b_kln[k, :, n] .- b_kln[k, k, n])
                    log_w_ikn[i, k, n] = - MDToolbox.logsumexp(x)
                end
            end
        end
    end
    
    return log_w_ikn
end

# k=1に対してΣ_i(exp(-f_ik))=1
function normalize_f_ik(f_ik)
    m, K = size(f_ik)
    normalized_f_ik = similar(f_ik)
    s = MDToolbox.logsumexp(-f_ik[:, 1])
    normalized_f_ik = f_ik .+ s
    return normalized_f_ik
end

# この関数は未完成
# 実際はlog_μ_xの計算で、x_gridをb_kで評価しなければならない
function tram_μ(f_ik, R_ik, b_k::Function, N_ik, index_of_cluster_kn, center_of_cluster, x_grid)
    m, K = size(f_ik)

    log_μ_x = zeros(Float64, length(x_grid))
    μ_x = zeros(Float64, length(x_grid))
    for x in x_grid
        ix = argmin((center_of_cluster .- x) .^ 2)[1]
        for k in 1:K
            log_μ_x = - MDToolbox.logsumexp(log(R_ik[ix, k]) + f_ik[ix, k] - b_k[k](x))
        end
    end

    μ_x = exp.(log_μ_x)
    return μ_x
end

function tram_init(c_ijk, N_ik)
    m, K = size(N_ik)
    f_ik = ones(Float64, m, K)
    f_ik = normalize_f_ik(f_ik)

    v_ik = ones(Float64, m, K)

    R_ik = tram_R_ik(c_ijk, v_ik, f_ik, N_ik)

    return f_ik, v_ik, R_ik
end