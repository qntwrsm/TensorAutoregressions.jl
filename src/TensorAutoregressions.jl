#=
TensorAutoregressions.jl

    Provides a collection of tools for working with tensor autoregressive 
    models, such as estimation, forecasting, and impulse response analysis. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/02/27
=#

module TensorAutoregressions

using LinearAlgebra

# Types
"""
    AbstractKruskal

Abstract type for Kruskal tensor.
"""
abstract type AbstractKruskal end

"""
    AbstractTensorErrorDistribution

Abstract type for tensor error distribution.
"""
abstract type AbstractTensorErrorDistribution end

# Structs
"""
    StaticKruskal <: AbstractKruskal

Static Kruskal tensor of rank ``R`` with loadings ``λ`` and factor matrices
``U``.
"""
mutable struct StaticKruskal{
    Tλ<:AbstractVector, 
    TU<:AbstractVector
} <: AbstractKruskal
    λ::Tλ
    U::TU
    R::Int
    function StaticKruskal(λ::AbstractVector, U::AbstractVector, R::Integer)
        length(unique(size.(U, 2))) == 1 || throw(DimensionMismatch("all factor matrices must have the same number of columns."))
        size(λ, 1) == size(U[1], 2) == R || throw(DimensionMismatch("number of loadings and number of columns of factor matrices must equal rank R."))

        return new{typeof(λ), typeof(U)}(λ, U, R)
    end
end

"""
    DynamicKruskal <: AbstractKruskal

Dynamic Kruskal tensor of rank ``R`` with dynamic loadings ``λ`` and factor
matrices ``U``. 
"""
mutable struct DynamicKruskal{
    Tλ<:AbstractMatrix,
    Tϕ<:AbstractMatrix,
    TΣ<:AbstractMatrix,
    TU<:AbstractVector
} <: AbstractKruskal
    λ::Tλ
    ϕ::Tϕ
    Σ::TΣ
    U::TU
    R::Int
    function DynamicKruskal(
        λ::AbstractVector, 
        ϕ::AbstractMatrix, 
        Σ::AbstractMatrix, 
        U::AbstractVector, 
        R::Integer
    )
        size(ϕ, 1) == size(ϕ, 2) == R || error("ϕ must be a square matrix of rank R.")
        issymmetric(Σ) && size(Σ, 1) == R || error("Σ must be a symmetric matrix of rank R.")
        length(unique(size.(U, 2))) == 1 || throw(DimensionMismatch("all factor matrices must have the same number of columns."))
        size(λ, 1) == size(U[1], 2) == R || throw(DimensionMismatch("number of loadings and number of columns of factor matrices must equal rank R."))

        return new{typeof(λ), typeof(ϕ), typeof(Σ), typeof(U)}(λ, ϕ, Σ, U, R)
    end
end

# helper functions
factors(A::AbstractKruskal) = A.U
loadings(A::AbstractKruskal) = A.λ
rank(A::AbstractKruskal) = A.R
function matricize(A::AbstractArray, n::Integer)
    sz = size(A)
    m = setdiff(1:ndims(A), n)
    perm = (n, n+1:ndims(A)..., 1:n-1...)

    return reshape(permutedims(A, perm), sz[n], prod(sz[m])) 
end
function tensorize(A::AbstractMatrix, n::Integer, sz::Tuple)
    m = setdiff(1:length(sz), n)
    perm = (length(sz)-(n-2):length(sz)..., 1:length(sz)-(n-1)...)

    return permutedims(reshape(A, sz[n], sz[m]...), perm)
end
function tucker(G::AbstractArray, A::AbstractVector)
    sz = size(G)
    for (i, Ai) ∈ enumerate(A)
        Gi = matricize(G, i)
        G = tensorize(Ai * Gi, i, size(G))
    end

    return G
end
full(A::AbstractKruskal) = kruskal(factors(A) .* loadings(A))
kruskal(A::AbstractVector) = tucker(I(length(A), size(A[1], 2)), A)
function (I::UniformScaling)(n::Integer, R::Integer)
    Id = zeros((n for _ = 1:R))
    for i = 1:R
        Id[i, i] = one(Float64)
    end

    return Id
end


"""
    WhiteNoise <: AbstractTensorErrorDistribution

White noise model of tensor errors ``ε`` with covariance matrix ``Σ`` of
``vec(ε)``.
"""
mutable struct WhiteNoise{
    Tε<:AbstractArray,
    TΣ<:AbstractMatrix
} <: AbstractTensorErrorDistribution
    ε::Tε
    Σ::TΣ
    function WhiteNoise(ε::AbstractArray, Σ::AbstractMatrix)
        issymmetric(Σ) || error("Σ must be symmetric.")
        prod(size(ε)[1:end-1]) == size(Σ, 1) || throw(DimensionMismatch("number of elements in vec(ε) must equal rank of Σ."))

        return new{typeof(ε), Symmetric}(ε, Symmetric(Σ))
    end
end

"""
    TensorNormal <: AbstractTensorErrorDistribution

Tensor normal model of tensor errors ``ε`` with separable covariance matrix
``Σ`` along each mode of the tensor.
"""
mutable struct TensorNormal{
    Tε<:AbstractArray,
    TΣ<:AbstractVector
} <: AbstractTensorErrorDistribution
    ε::Tε
    Σ::TΣ
    function TensorNormal(ε::AbstractArray, Σ::AbstractVector)
        length(Σ) == ndims(ε) - 1 || error("number of matrices in Σ must equal number of modes of ε.")
        all(size.(Σ, 1) .== size(ε)[1:end-1]) || throw(DimensionMismatch("dimensions of matrices in Σ must equal dimensions of modes of ε."))
        all(issymmetric, Σ) || error("all matrices in Σ must be symmetric.")

        return new{typeof(ε), Symmetric}(ε, Symmetric.(Σ))
    end
end

# helper functions
resid(ε::AbstractTensorErrorDistribution) = ε.ε
cov(ε::AbstractTensorErrorDistribution) = ε.Σ

"""
    TensorAutoregression

Tensor autoregressive model with tensor error distribution ``ϵ`` and Kruskal tensor
representation ``A``, potentially dynamic.
"""
mutable struct TensorAutoregression{
    Ty<:AbstractArray, 
    Tε<:AbstractTensorErrorDistribution,
    TA<:AbstractKruskal
}
    y::Ty
    ε::Tε
    A::TA
    function TensorAutoregression(y::AbstractArray, ε::AbstractTensorErrorDistribution, A::AbstractKruskal)
        size(y) == size(resid(ε)) || throw(DimensionMismatch("dimensions of y and residuals must be equal."))
        all(size(y)[1:end-1] .== size.(factors(A), 1)) || throw(DimensionMismatch("dimensions of loadings must equal number of columns of y."))

        return new{typeof(y), typeof(ε), typeof(A)}(y, ε, A)
    end
end

end
