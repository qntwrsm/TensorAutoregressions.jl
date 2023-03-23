#=
utilities.jl

    Provides a collection of utility tools for working with tensor 
    autoregressive models, such as moving average representation, orthogonalize 
    responses, and state space form for the dynamic model. 

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
    state_space(y, A, ε) -> sys

State space form of the tensor autoregressive model with dynamic Kruskal
coefficient tensor `A` and tensor error distribution `ε`.
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
    Z = [inv(norm(Xt)) for Xt in eachslice(X, dims=n+1)]
    A_star = tucker(X, Cinv', 1:n)
    y_star = [inv(Z[t]) * dot(vec(selectdim(A_star, n+1, t)), vec(selectdim(y, n+1, t+1))) for t = 1:last(dims)-1]

    # system
    sys = LinearTimeVariant(
        y_star,
        Z,
        [dynamics(A) for _ = 1:last(dims)-1],
        zero(y_star),
        zero(y_star),
        [Matrix{eltype(y_star)}(I, rank(A), rank(A)) for _ = 1:last(dims)-1],
        [cov(A) for _ = 1:last(dims)-1],
        zeros(eltype(y_star), rank(A)),
        Matrix{eltype(y_star)}(I, rank(A), rank(A))
    )

    return sys
end