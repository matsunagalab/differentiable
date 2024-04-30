"""
    tram(c_ijk, N_ik, b_kln, index_of_cluster_kn; tol=1e-10, max_iter=20)

Estimates the free energy differences by transition-based reweighting analysis method (TRAM).

## Arguments
- `c_ijk::AbstractArray`: Array containing the counts of transitions from state `i` to state `j` in ensemble `k`.
- `N_ik::AbstractArray`: Array containing the counts of samples in state `i` in ensemble `k`.
- `N_k::AbstractArray`: Array containing the counts of samples in ensemble `k`.
- `b_kln::AbstractArray`: Array of bias factors of state `k` evaluated at state `l`.
- `index_of_cluster_kn::AbstractArray`: Array fo index of the state to which the nth sample of ensemble `k` belongs.. 

## Optional Arguments
- `tol::Real = 1e-10`: Tolerance value for convergence criteria.
- `max_iter::Integer = 20`: Maximum number of iterations allowed.
"""
function tram(c_ijk, N_ik, N_k, b_kln, index_of_cluster_kn; tol=1e-10, max_iter=100)
    f_ik, v_ik, R_ik = tram_init(c_ijk, N_ik)

    for i in 1:max_iter
        for j in 1:100 #この100はなんとなくの値, f_ikの更新に比べ計算量が少ないため適当に大きい値を設定している
            v_new_ik = tram_v_new_ik(c_ijk, v_ik, f_ik)
            #=
            if(j == 100)
                println("delta = $(maximum(abs.(v_ik .- v_new_ik)))")
                v_ik = v_new_ik
            end
            =#
            v_ik = v_new_ik
        end
        
        R_ik = tram_R_ik(c_ijk, v_ik, f_ik, N_ik)
        f_new_ik = tram_f_new_ik(f_ik, R_ik, b_kln, N_k, index_of_cluster_kn)
        max_dif = maximum(abs.(f_ik .- f_new_ik))
        if i % (max_iter/10) == 1 || i == max_iter 
            println("iteration = $(i), delta = $(max_dif)")
        end
        f_ik = f_new_ik
        if(max_dif < tol)
            println("iteration = $(i), delta = $(max_dif)")
            break
        end
    end
    
    return f_ik, v_ik, R_ik
end

# 複数の返り値を持つ関数のrruleの書き方が分からないため、適当な関数を用意する
function tram_f(f_ik, R_ik, b_kln, N_ik, N_k, index_of_cluster_kn)
    return f_ik
end

function ChainRulesCore.rrule(::typeof(tram_f), f_ik, R_ik, b_kln, N_ik, N_k, index_of_cluster_kn)
    m, K = size(f_ik)
    f_ik = tram(f_ik, b_kln)
    function tram_f_pullback(df)
        db_kln = similar(b_kln)
        w_ik = Array{Array{Float64}}(undef, m, K)
        for k in 1:K
            for i in 1:m
                log_wik_jn = tram_log_wik_jn(f_ik, R_ik, b_kln, N_k, i, k)
                w_ik[i, k] = exp.(log_wik_jn)
            end
        end
        for k in 1:K
            for l in 1:K
                for n in 1:N_k[k]
                    istate = index_of_cluster_kn[k, n]
                    db_kln[k, l, n] = - w_ik[istate, k][l, n] / f_ik[i, k] * df[i, k]
                end
            end
        end
        return NoTangent(), NoTangent(), NoTangent(), db_kln, NoTangent(), NoTangent(), NoTangent()
    end

    return f_ik, tram_f_pullback
end

function tram_by_delta(c_ijk, N_ik, N_k, b_kln, index_of_cluster_kn; tol=1e-10, max_iter=10000)
    f_ik, v_ik, R_ik = tram_init(c_ijk, N_ik)

    for i in 1:max_iter
        for j in 1:100 #この100はなんとなくの値, f_ikの更新に比べ計算量が少ないため適当に大きい値を設定している
            v_new_ik = tram_v_new_ik(c_ijk, v_ik, f_ik)
            #=
            if(j == 100)
                println("delta = $(maximum(abs.(v_ik .- v_new_ik)))")
                v_ik = v_new_ik
            end
            =#
            v_ik = v_new_ik
        end
        
        R_ik = tram_R_ik(c_ijk, v_ik, f_ik, N_ik)
        #f_new_ik = tram_f_new_ik(f_ik, R_ik, b_kln, N_k, index_of_cluster_kn)
        f_new_ik = similar(f_ik)
        δ_i = tram_δ_i(c_ijk, v_ik, f_ik)
        for j in 1:m
            f_new_ik[j, :] = f_ik[j, :] .+ δ_i[j]
        end
        
        max_dif = maximum(abs.(f_ik .- f_new_ik))
        if i % (max_iter/10) == 1 || i == max_iter 
            println("iteration = $(i), delta = $(max_dif)")
        end
        f_ik = f_new_ik
        if(max_dif < tol)
            println("iteration = $(i), delta = $(max_dif)")
            break
        end
    end
    
    return f_ik, v_ik, R_ik
end

function tram_δ_i(c_ijk, v_ik, f_ik)
    m, K = size(v_ik)

    δ_i = zeros(Float64, m)
    for i in 1:m
        x = 0
        for k in 1:K
            for j in 1:m
                x += (c_ijk[i, j, k] + c_ijk[j, i, k]) * v_ik[j, k] / (v_ik[j, k] + exp(f_ik[i, k] - f_ik[j, k]) * v_ik[i, k])
            end
        end
        δ_i[i] = log(x) - log(sum(c_ijk[:, i, :]))
    end

    return δ_i
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

function ChainRulesCore.rrule(::typeof(tram_p_ijk), c_ijk, v_ik, f_ik)
    m, K = size(f_ik)
    p_ijk = tram_p_ijk(c_ijk, v_ik, f_ik)

    function tram_p_ijk_pullback(dp)
        df_ik = similar(f_ik)
        for k in 1:K
            for i in 1:m
                for j in 1:m
                    df_ik[i, k] += p_ijk[i, j, k] * (1 - v_ik[i, k] / (exp(f_ik[j, k] - f_ik[i, k]) * v_ik[j, k] + v_ik[i, k])) * dp[i, j, k]
                end
            end
        end
        return NoTangent(), NoTangent(), NoTangent(), df_ik
    end
    return p_ijk, tram_p_ijk_pullback
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
function tram_f_new_ik(f_ik, R_ik, b_kln, N_k, index_of_cluster_kn)
    m, K = size(f_ik)
    f_new_ik = zeros(Float64, m, K)

    for k in 1:K
        for i in 1:m
            log_wik_jn = tram_log_wik_jn(f_ik, R_ik, b_kln, N_k, i, k)
            log_wik = log_wik_jn[index_of_cluster_kn .== i]
            f_new_ik[i, k] = - MDToolbox.logsumexp(log_wik)
        end
    end

    f_new_ik = normalize_f_ik(f_new_ik)
    return f_new_ik
end

# K×N_maxの配列で、f_ikの更新に用いる
# j行目n列目はjアンサンブルのn個目のデータを用い更新する
function tram_log_wik_jn(f_ik, R_ik, b_kln, N_k, istate, kstate)
    m, K = size(f_ik)
    log_wik_jn = zeros(Float64, (K, maximum(N_k)))

    for k in 1:K
        x = repeat(log.(R_ik[istate, :]), 1, N_k[k]) .+ repeat(f_ik[istate, :], 1, N_k[k]) .- (b_kln[k, :, 1:N_k[k]] .- repeat(b_kln[k:k, kstate, 1:N_k[k]], K, 1))
        log_wik_jn[k:k, 1:N_k[k]] .= - logsumexp_over_row(x)
    end
    return log_wik_jn
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

function logsumexp_over_row(x)
    max_x = maximum(x, dims=1)
    exp_x = exp.(x .- max_x)
    s = log.(sum(exp_x, dims=1)) .+ max_x
    return s
end