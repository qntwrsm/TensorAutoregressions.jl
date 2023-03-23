#=
utilities.jl

    Provides a collection of utility tools for working with tensor 
    autoregressive models, such as moving average representation, orthogonalize 
    responses, state space form for the dynamic model, as well as filter and 
    smoother routines. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/07/02
=#

"""
    moving_average(A, n) -> Ψ

Moving average, ``MA(∞)``, representation of the tensor autoregressive model
with Kruskal coefficient tensor `A`, computed up to the `n`th term.
"""
function moving_average(A::StaticKruskal, n::Integer)
    dims = size(A)

    # matricize Kruskal tensor
    An = matricize(full(A), 1:length(dims)÷2)

    # moving average coefficients
    Ψ = zeros(dims..., n+1)
    for h = 0:n
        selectdim(Ψ, ndims(Ψ), h+1) .= tensorize(An^h, 1:length(dims)÷2, dims)
    end

    return Ψ
end

function moving_average(A::DynamicKruskal, n::Integer)
    # TODO: implementation
    error("moving average representation not implemented for dynamic model.")
end

"""
    orthogonalize(Ψ, Σ) -> Ψ_orth

Orthogonalize impulse responses `Ψ` using the Cholesky decomposition of
covariance matrix `Σ`.
"""
function orthogonalize(Ψ::AbstractArray, Σ::AbstractMatrix)
    # Cholesky decomposition of Σ
    C = cholesky(Hermitian(Σ))

    # orthogonalize responses
    Ψ_orth = similar(Ψ)
    for (h, ψ) ∈ pairs(eachslice(Ψ, dims=ndims(Ψ)))
        selectdim(Ψ_orth, ndims(Ψ_orth), h) .= tensorize(matricize(ψ, 1:ndims(ψ)÷2) * C.L, 1:ndims(ψ)÷2, size(ψ))
    end

    return Ψ_orth
end

function orthogonalize(Ψ::AbstractArray, Σ::AbstractVector)
    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(Σ))

    # orthogonalize responses
    Ψ_orth = similar(Ψ)
    for (h, ψ) ∈ pairs(eachslice(Ψ, dims=ndims(Ψ)))
        selectdim(Ψ_orth, ndims(Ψ_orth), h) .= tucker(ψ, C)
    end
    return Ψ_orth
end

"""
    state_space(y, A, ε) -> (y_star, Z_star, a1, P1)

State space form of the collapsed tensor autoregressive model with dynamic
Kruskal coefficient tensor `A` and tensor error distribution `ε`.
"""
function state_space(y::AbstractArray, A::DynamicKruskal, ε::TensorNormal)
    dims = size(y)
    n = ndims(y) - 1

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(ε)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # scaling
    S = [Cinv[i] * U[i] for i = 1:n]
    
    # collapsing
    X = tucker(selectdim(y, n+1, 1:last(dims)-1), S, 1:n)
    Z_star = [inv(norm(Xt)) for Xt in eachslice(X, dims=n+1)]
    A_star = tucker(X, Cinv', 1:n)
    y_star = [inv(Z[t]) * dot(vec(selectdim(A_star, n+1, t)), vec(selectdim(y, n+1, t+1))) for t = 1:last(dims)-1]

    # initial conditions
    a1 = zeros(eltype(y_star), rank(A))
    P1 = Matrix{eltype(y_star)}(rank(A), rank(A))

    return (y_star, Z_star, a1, P1)
end

"""
    filter(y, Z, T, Q, a1, P1) -> (a, P, v, F, K)

Collapsed Kalman filter for the dynamic tensor autoregressive model with system
matrices `Z`, `T`, and `Q` and initial conditions `a1` and `P1`. Returns the
filtered state `a`, covariance `P`, forecast error `v`, forecast error variance
`F`, and Kalman gain `K`.
"""
function filter(
    y::AbstractVector, 
    Z::AbstractVector, 
    T::AbstractMatrix, 
    Q::AbstractMatrix, 
    a1::AbstractVector, 
    P1::AbstractMatrix
)
    a = similar(y, typeof(a1))
    P = similar(y, typeof(P1))
    v = similar(y)
    F = similar(y, typeof(P1))
    K = similar(y, typeof(P1))

    # initialize filter
    a[1] = a1
    P[1] = P1

    # filter
    for t ∈ eachindex(y)
        # forecast error
        v[t] = y[t] - Z[t] * a[t]
        F[t] = Z[t] * P[t] * Z[t]'

        # Kalman gain
        K[t] = T * P[t] * Z[t]' / F[t]

        # prediction
        a[t+1] = T * a[t] + K[t] * v[t]
        P[t+1] = T * P[t] * (T - K[t] * Z[t])' + Q
    end

    return (a, P, v, F, K)
end

"""
    smoother(y, Z, T, Q, a1, P1) -> (α̂, V, Γ)

Collapsed Kalman smoother for the dynamic tensor autoregressive model with
system matrices `Z`, `T`, and `Q` and initial conditions `a1` and `P1`. Returns
the smoothed state `α̂`, covariance `V`, and autocovariance `Γ`.
"""
function smoother(
    y::AbstractVector, 
    Z::AbstractVector, 
    T::AbstractMatrix, 
    Q::AbstractMatrix, 
    a1::AbstractVector, 
    P1::AbstractMatrix,
)
    # filter
    (a, P, v, F, K) = filter(y, Z, T, Q, a1, P1)

    α̂ = similar(a)
    V = similar(P)
    Γ = similar(P, length(y)-1)

    # initialize smoother
    r = zero(a1)
    N = zero(P1)
    L = similar(T)

    # smoother
    for y ∈ reverse(eachindex(y))
        L .= T - K[t] * Z[t]

        # backward recursion
        r .= Z[t]' / F[t] * v[t] + L' * r
        N .= Z[t]' / F[t] * Z[t] + L' * N * L

        # smoothing
        α̂[t] = a[t] + P[t] * r
        V[t] = P[t] - P[t] * N * P[t]
        t > 1 && Γ[t-1] = I - P[t] * N
        t < length(y) && Γ[t] *= L * P[t]
    end

    return (α̂, V, Γ)
end