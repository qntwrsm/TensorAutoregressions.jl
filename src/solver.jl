#=
solver.jl

    Provides alternating least squares (ALS) and expectation-maximization (EM) 
    optimization routines for fitting a tensor autoregressive model with Kruskal 
    coefficient tensor and tensor error distribution. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/08/02
=#

"""
    update_factor!(u, w, P, scale)

Update and normalize factor `u` using regression based on projection matrix `P`
and companion factor `w`.
"""
function update_factor!(u::AbstractVector, w::AbstractVector, P::AbstractMatrix, scale::Real)
    # update factor
    mul!(u, P, w, scale, zero(eltype(u)))
    # normalize
    u .*= inv(norm(u)) 

    return nothing
end

"""
    update_dynamic!(A, σ̂, γ̂)

Update dynamics of dynamic Kruskal coefficient tensor `A` using smoothed
loadings variance `σ̂`, and autocovariance `γ̂`.
"""
function update_dynamic!(A::DynamicKruskal, σ̂::AbstractVector, γ̂::AbstractVector)
    # lags and leads
    λ̂_lag = @view loadings(A)[1:end-1]
    λ̂_lead = @view loadings(A)[2:end]
    σ̂_lag = @view σ̂[1:end-1]
    σ̂_lead = @view σ̂[2:end]

    # second moments
    φ_lag = σ̂_lag + abs2.(λ̂_lag)
    φ_lead = σ̂_lead + abs2.(λ̂_lead)
    φ_cross = γ̂ + σ̂_lead .* σ̂_lag

    # update dynamics
    dynamics(A) = sum(φ_cross) * inv(sum(φ_lag))
    cov(A) = inv(length(loadings(A)) - 1) * (sum(φ_lead) - sum(x -> abs2(x[1]) * inv(x[2]), zip(φ_cross, φ_lag)))

    return nothing
end

"""
    update_static!(A, ε, y. σ̂)

Update static factors of dynamic Kruskal coefficient tensor `A` and tensor error
distribution `ε` based on data `y` and smoothed loading variance `σ̂`.
"""
function update_static!(A::DynamicKruskal, ε::TensorNormal, y::AbstractArray, σ̂::AbstractVector)
    dims = size(y)
    n = ndims(y) - 1

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(ε)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]
    # precision matrices Ωᵢ
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # scaling
    S = [Cinv[i].L * U[i] for i = 1:n]

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

    # smoother variables
    φ = σ̂ + abs2.(loadings(A))

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
        λ̂_ext = repeat(loadings(A), inner=prod(dims[m]))
        φ_ext = repeat(φ, inner=prod(dims[m]))

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
            inv(G) * M' * Ω[k], 
            inv(dot(factors(A)[k], Ω[k], factors(A)[k]))
        )

        # update outer product of Kruskal factors
        U[k] = factors(A)[k] * factors(A)[k+n]'

        # update covariance
        μk = U[k] * Xk
        Ek = Zk - λ̂_ext' .* μk
        mul!(cov(ε).data[k], Ek, Ek', inv((last(dims) - 1) * prod(dims[m])), .0)
        cov(ε).data[k] += inv((last(dims) - 1) * prod(dims[m])) .* σ̂_ext' .* μk * μk' 

        # update scaling
        S[k] = Cinv[k].L * U[k]
    end

    # update residuals
    resid(ε) .= y_lead .- reshape(loadings(A), ones(Int, n)..., :) .* tucker(y_lag, U, 1:n)

    return nothing
end

"""
    estep!(A, y, ε) -> (σ̂, γ̂)

Compute smoothed loadings, loadings variance `σ̂` and autocovariance `γ̂` as
part of the expectation step (E-step) of the EM algorithm for fitting a dynamic
tensor autoregressive model to the data `y` by means of the collapsed Kalman
filter and smoother routines.
"""
function estep(A::DynamicKruskal, y::AbstractArray, ε::TensorNormal)
    dims = size(y)
    n = ndims(y) - 1

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(ε)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # scaling
    S = [Cinv[i].L * U[i] for i = 1:n]

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)
    
    # collapsing
    X = tucker(y_lag, S, 1:n)
    Z = [[inv(norm(Xt))] for Xt in eachslice(X, dims=n+1)]

    # smoother
    (λ̂, σ̂, γ̂) = smoother(y_lead, Z, )
    loadings(A) .= λ̂

    return (σ̂, γ̂)
end

"""
    update!(A, ε, y)

Update Kruskal coefficient tensor `A` and tensor error distribution `ε` for the
tensor autoregressive model based on data `y`.
"""
function update!(A::StaticKruskal, ε::WhiteNoise, y::AbstractArray)
    dims = size(y)
    n = ndims(y) - 1

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

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
        update_factor!(factors(A)[k+n], factors(A)[k], inv(G) * M', inv(loadings(A)[1]))

        # update outer product of Kruskal factors
        U[k] = factors(A)[k] * factors(A)[k+n]'
    end

    # regressor tensor
    X = tucker(y_lag, U, 1:n)

    # update loading
    loadings(A)[1] = dot(y_lead, X) * inv(norm(X)^2)

    # update residuals
    resid(ε) .= y_lead .- loadings(A)[1] .* X

    # update covariance
    E = matricize(resid(ε), 1:n)
    mul!(cov(ε).data, E, E')

    return nothing
end

function update!(A::StaticKruskal, ε::TensorNormal, y::AbstractArray)
    dims = size(y)
    n = ndims(y) - 1

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(ε)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]
    # precision matrices Ωᵢ
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # scaling
    S = [Cinv[i].L * U[i] for i = 1:n]

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
            inv(G) * M' * Ω[k], 
            inv(loadings(A)[1] * dot(factors(A)[k], Ω[k], factors(A)[k]))
        )

        # update outer product of Kruskal factors
        U[k] = factors(A)[k] * factors(A)[k+n]'

        # update covariance
        Ek = Zk - loadings(A)[1] .* Uk * Xk
        mul!(cov(ε).data[k], Ek, Ek', inv((last(dims) - 1) * prod(dims[m])), .0)
    
        # update scaling
        S[k] = Cinv[k].L * U[k]
    end

    # dependent variable and regressor tensors
    Z = tucker(y_lead, Cinv, 1:n)
    X = tucker(y_lag, S, 1:n)

    # update loading
    loadings(A)[1] = dot(Z, X) * inv(norm(X)^2)

    # update residuals
    resid(ε) .= y_lead .- loadings(A)[1] .* tucker(y_lag, U, 1:n)

    return nothing
end

function update!(A::DynamicKruskal, ε::TensorNormal, y::AbstractArray)
    # E-step
    (σ̂, γ̂) = estep!(A, y, ε)

    # M-step
    update_dynamic!(A, σ̂, γ̂)
    update_static!(A, ε, y, σ̂)

    return nothing
end