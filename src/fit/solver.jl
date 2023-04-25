#=
solver.jl

    Provides alternating least squares (ALS) and expectation-maximization (EM) 
    optimization routines for fitting a tensor autoregressive model with Kruskal 
    coefficient tensor and tensor error distribution. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/08/02
=#

"""
    update!(A, ε, y, fixed)

Update Kruskal coefficient tensor `A` and tensor error distribution `ε` for the
tensor autoregressive model based on data `y`, with fixed parameters indicated
by `fixed`.
"""
function update!(A::StaticKruskal, ε::WhiteNoise, y::AbstractArray, fixed::NamedTuple)
    dims = size(y)
    n = ndims(y) - 1

    # extract fixed parameters
    fixed_coef = get(fixed, :coef, NamedTuple())

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

    if !haskey(fixed_coef, :factors)
        # loop through modes
        for k = 1:n
            m = setdiff(1:n, k)
            # matricize dependent variable along k-th mode
            Yk = matricize(y_lead, k)
            # matricize regressor along k-th mode
            X = tucker(y_lag, U[m], m)
            Xk = matricize(X, k)

            # Gram matrix
            G = Xk * Xk'
            # moment matrix
            M = Yk * Xk'

            # update factor k
            update_factor!(
                factors(A)[k], 
                factors(A)[k+n], 
                M, 
                inv(loadings(A)[1] * dot(factors(A)[k+n], G, factors(A)[k+n]))
            )
            # update factor k+n
            update_factor!(factors(A)[k+n], factors(A)[k], G \ M', inv(loadings(A)[1]))

            # update outer product of Kruskal factors
            U[k] = factors(A)[k] * factors(A)[k+n]'
        end
    end

    # regressor tensor
    X = tucker(y_lag, U, 1:n)

    # update loading
    if !haskey(fixed_coef, :loadings)
        f(x) = norm(y_lead)^2 - 2 * dot(y_lead, X) * x + norm(X)^2 * x^2 
        res = optimize(f, -1.0, 1.0)
        loadings(A)[1] = Optim.minimizer(res)
    end

    # update residuals
    resid(ε) .= y_lead .- loadings(A)[1] .* X

    # update covariance
    E = matricize(resid(ε), 1:n)
    mul!(cov(ε).data, E, E')

    return nothing
end

function update!(A::StaticKruskal, ε::TensorNormal, y::AbstractArray, fixed::NamedTuple)
    dims = size(y)
    n = ndims(y) - 1

    # extract fixed parameters
    fixed_coef = get(fixed, :coef, NamedTuple())
    fixed_dist = get(fixed, :dist, NamedTuple())

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(ε)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]
    # precision matrices Ωᵢ
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # scaling
    S = [Cinv[i] * U[i] for i = 1:n]

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

    for k = 1:n
        m = setdiff(1:n, k)
        # matricize dependent variable along k-th mode
        Z = tucker(y_lead, Cinv[m], m)
        Zk = matricize(Z, k)
        # matricize regressor along k-th mode
        X = tucker(y_lag, S[m], m)
        Xk = matricize(X, k)

        if !haskey(fixed_coef, :factors)
            # Gram matrix
            G = Xk * Xk'
            # moment matrix
            M = Zk * Xk'

            # update factor k
            update_factor!(
                factors(A)[k], 
                factors(A)[k+n], 
                M, 
                inv(loadings(A)[1] * dot(factors(A)[k+n], G, factors(A)[k+n]))
            )
            # update factor k+n
            update_factor!(
                factors(A)[k+n], 
                factors(A)[k], 
                G \ M' * Ω[k], 
                inv(loadings(A)[1] * dot(factors(A)[k], Ω[k], factors(A)[k]))
            )

            # update outer product of Kruskal factors
            U[k] = factors(A)[k] * factors(A)[k+n]'
        end

        if !haskey(fixed_dist, :cov)
            # update covariance
            Ek = Zk - loadings(A)[1] .* U[k] * Xk
            mul!(cov(ε)[k].data, Ek, Ek', inv((last(dims) - 1) * prod(dims[m])), .0)
            # normalize
            k != n && lmul!(inv(norm(cov(ε)[k])), cov(ε)[k].data)

            # update Cholesky decomposition
            Cinv[k] = inv(cholesky(Hermitian(cov(ε)[k])).L)

            # update precision matrix
            Ω[k] = transpose(Cinv)[k] .* Cinv[k]
        end
    
        # update scaling
        S[k] = Cinv[k] * U[k]
    end

    # dependent variable and regressor tensors
    Z = tucker(y_lead, Cinv, 1:n)
    X = tucker(y_lag, S, 1:n)

    # update loading
    if !haskey(fixed_coef, :loadings)
        f(x) = norm(Z)^2 - 2 * dot(Z, X) * x + norm(X)^2 * x^2 
        res = optimize(f, -1.0, 1.0)
        loadings(A)[1] = Optim.minimizer(res)
    end

    # update residuals
    resid(ε) .= y_lead .- loadings(A)[1] .* tucker(y_lag, U, 1:n)

    return nothing
end

function update!(A::DynamicKruskal, ε::TensorNormal, y::AbstractArray, fixed::NamedTuple)
    # E-step
    # collapsed state space system
    (y_star, Z_star, a1, P1) = state_space(y, A, ε)
    # smoother
    (α̂, V, Γ) = smoother(y_star, Z_star, intercept(A), dynamics(A), cov(A), a1, P1)
    loadings(A) .= hcat(α̂...)
    σ̂ = vec(vcat(V...))
    γ̂ = vec(vcat(Γ...))

    # M-step
    update_dynamic!(A, σ̂, γ̂, get(fixed, :coef, NamedTuple()))
    update_static!(A, ε, y, σ̂, fixed)

    return nothing
end

"""
    update_factor!(u, w, P, scale)

Update and normalize factor `u` using regression based on projection matrix `P`
and companion factor `w`.
"""
function update_factor!(u::AbstractVecOrMat, w::AbstractVecOrMat, P::AbstractMatrix, scale::Real)
    # update factor
    mul!(u, P, w, scale, zero(eltype(u)))
    # normalize
    u .*= inv(norm(u)) 

    return nothing
end

"""
    update_dynamic!(A, σ̂, γ̂, fixed)

Update dynamics of dynamic Kruskal coefficient tensor `A` using smoothed
loadings variance `σ̂`, and autocovariance `γ̂`, with fixed parameters indicated
by `fixed`.
"""
function update_dynamic!(
    A::DynamicKruskal, 
    σ̂::AbstractVector, 
    γ̂::AbstractVector,
    fixed::NamedTuple
)
    # lags and leads
    λ̂_lag = @view loadings(A)[1:end-1]
    λ̂_lead = @view loadings(A)[2:end]
    σ̂_lag = @view σ̂[1:end-1]
    σ̂_lead = @view σ̂[2:end]

    # second moments
    φ_lag = σ̂_lag + abs2.(λ̂_lag)
    φ_lead = σ̂_lead + abs2.(λ̂_lead)
    φ_cross = γ̂ + λ̂_lead .* λ̂_lag

    # objective closures
    f_intercept(x) = x^2 - 2 * (mean(λ̂_lead) - dynamics(A)[1] * mean(λ̂_lag)) * x
    function f_dynamics(x)
        scale = one(x) - x^2
        α = intercept(A)[1]
        c = mean(φ_lead) + α^2 - 2 * α * mean(λ̂_lead)
        f = c + 2 * (α * mean(λ̂_lag) - mean(φ_cross)) * x + mean(φ_lag) * x^2

        return log(scale) + f * inv(scale)
    end

    # update dynamics
    if !haskey(fixed, :dynamics)
        res = optimize(f_dynamics, 0.0, 1.0)
        dynamics(A) .= Optim.minimizer(res)
    end
    if !haskey(fixed, :intercept)
        res = optimize(f_intercept, 0.0, 0.7 * (1 - dynamics(A)[1]))
        intercept(A) .= Optim.minimizer(res)
    end
    cov(A).data .= I - dynamics(A) * dynamics(A)'

    return nothing
end

"""
    update_static!(A, ε, y. σ̂, fixed)

Update static factors of dynamic Kruskal coefficient tensor `A` and tensor error
distribution `ε` based on data `y` and smoothed loading variance `σ̂`, with
fixed parameters indicated by `fixed`.
"""
function update_static!(
    A::DynamicKruskal, 
    ε::TensorNormal, 
    y::AbstractArray, 
    σ̂::AbstractVector,
    fixed::NamedTuple
)
    dims = size(y)
    n = ndims(y) - 1

    # extract fixed parameters
    fixed_coef = get(fixed, :coef, NamedTuple())
    fixed_dist = get(fixed, :dist, NamedTuple())

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(ε)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]
    # precision matrices Ωᵢ
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # scaling
    S = [Cinv[i] * U[i] for i = 1:n]

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

    # smoother variables
    φ = σ̂ + abs2.(vec(loadings(A)))

    for k = 1:n
        m = setdiff(1:n, k)
        # matricize dependent variable along k-th mode
        Z = tucker(y_lead, Cinv[m], m)
        Zk = matricize(Z, k)
        # matricize regressor along k-th mode
        X = tucker(y_lag, S[m], m)
        Xk = matricize(X, k)

        # repeat/extent smoother variables
        σ̂_ext = repeat(σ̂, inner=prod(dims[m]))
        λ̂_ext = repeat(vec(loadings(A)), inner=prod(dims[m]))
        φ_ext = repeat(φ, inner=prod(dims[m]))

        if !haskey(fixed_coef, :factors)
            # Gram matrix
            G = φ_ext' .* Xk * Xk'
            # moment matrix
            M = λ̂_ext' .* Zk * Xk'

            # update factor k
            update_factor!(
                factors(A)[k], 
                factors(A)[k+n], 
                M, 
                inv(dot(factors(A)[k+n], G, factors(A)[k+n]))
            )
            # update factor k+n
            update_factor!(
                factors(A)[k+n], 
                factors(A)[k], 
                G \ M' * Ω[k], 
                inv(dot(factors(A)[k], Ω[k], factors(A)[k]))
            )

            # update outer product of Kruskal factors
            U[k] = factors(A)[k] * factors(A)[k+n]'
        end

        if !haskey(fixed_dist, :cov)
            # update covariance
            μk = U[k] * Xk
            Ek = Zk - λ̂_ext' .* μk
            mul!(cov(ε)[k].data, Ek, Ek', inv((last(dims) - 1) * prod(dims[m])), .0)
            cov(ε)[k].data .+= inv((last(dims) - 1) * prod(dims[m])) .* σ̂_ext' .* μk * μk'
            # normalize
            k != n && lmul!(inv(norm(cov(ε)[k])), cov(ε)[k].data)

            # update Cholesky decomposition
            Cinv[k] = inv(cholesky(Hermitian(cov(ε)[k])).L)

            # update precision matrix
            Ω[k] = transpose(Cinv)[k] .* Cinv[k]
        end

        # update scaling
        S[k] = Cinv[k] * U[k]
    end

    # update residuals
    resid(ε) .= y_lead .- reshape(loadings(A), ones(Int, n)..., :) .* tucker(y_lag, U, 1:n)

    return nothing
end