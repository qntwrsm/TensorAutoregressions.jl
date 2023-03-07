#=
utilities.jl

    Provides a collection of utility tools for working with tensor 
    autoregressive models, such as moving average representation and 
    orthogonalize responses. 

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
