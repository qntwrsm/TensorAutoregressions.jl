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
        U[k] .= factors(A)[k] * factors(A)[k+n]'
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
        U[k] .= factors(A)[k] * factors(A)[k+n]'

        # update residuals
        resid(ε) .= Z .- loadings(A)[1] * tucker(X, U[k], k)

        # update covariance
        Ek = matricize(resid(ε), k)
        mul!(cov(ε).data[k], Ek, Ek', inv((last(dims) - 1) * prod(dims[m])), .0)
    
        # update scaling
        S[k] .= Cinv[k].L * U[k]
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