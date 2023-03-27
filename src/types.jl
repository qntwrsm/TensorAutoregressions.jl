#=
types.jl

    Provides a collection of types for working with tensor autoregressive 
    models, such as the Kruskal tensor for the autoregressive coefficients and 
    tensor error distributions, as well as the main tensor autoregressive model 
    itself. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/03/02
=#

# Autoregrssive coefficient Kruskal tensor
"""
    AbstractKruskal

Abstract type for Kruskal tensor.
"""
abstract type AbstractKruskal end

"""
    StaticKruskal <: AbstractKruskal

Static Kruskal tensor of rank `R` with loadings `λ` and factor matrices `U`.
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
        length(λ) == size(U[1], 2) == R || throw(DimensionMismatch("number of loadings and number of columns of factor matrices must equal rank R."))

        return new{typeof(λ), typeof(U)}(λ, U, R)
    end
end

"""
    DynamicKruskal <: AbstractKruskal

Dynamic Kruskal tensor of rank `R` with dynamic loadings `λ` and factor matrices
`U`. 
"""
mutable struct DynamicKruskal{
    Tλ<:AbstractVecOrMat,
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
        λ::AbstractVecOrMat, 
        ϕ::AbstractMatrix, 
        Σ::AbstractMatrix, 
        U::AbstractVector, 
        R::Integer
    )
        size(ϕ, 1) == size(ϕ, 2) == R || error("ϕ must be a square matrix of rank R.")
        issymmetric(Σ) && size(Σ, 1) == R || error("Σ must be a symmetric matrix of rank R.")
        length(unique(size.(U, 2))) == 1 || throw(DimensionMismatch("all factor matrices must have the same number of columns."))
        size(λ, 2) == size(U[1], 2) == R || throw(DimensionMismatch("number of loadings and number of columns of factor matrices must equal rank R."))

        return new{typeof(λ), typeof(ϕ), typeof(Σ), typeof(U)}(λ, ϕ, Σ, U, R)
    end
end

# methods
factors(A::AbstractKruskal) = A.U
loadings(A::AbstractKruskal) = A.λ
rank(A::AbstractKruskal) = A.R
Base.size(A::AbstractKruskal) = tuple(size.(factors(A), 1)...)
full(A::AbstractKruskal) = tucker(loadings(A) .* I(length(factors(A)), rank(A)), factors(A))
dynamics(A::DynamicKruskal) = A.ϕ
Statistics.cov(A::DynamicKruskal) = A.Σ
Base.similar(A::StaticKruskal) = StaticKruskal(similar(loadings(A)), similar.(factors(A)), rank(A))
Base.similar(A::DynamicKruskal) = DynamicKruskal(similar(loadings(A)), similar(dynamics(A)), similar(cov(A)), similar.(factors(A)), rank(A))
function Base.copyto!(dest::StaticKruskal, src::StaticKruskal)
    copyto!(loadings(dest), loadings(src))
    copyto!.(factors(dest), factors(src))
    rank(dest) = rank(src)

    return dest
end
function Base.copyto!(dest::DynamicKruskal, src::DynamicKruskal)
    copyto!(loadings(dest), loadings(src))
    copyto!(dynamics(dest), dynamics(src))
    copyto!(cov(dest), cov(src))
    copyto!.(factors(dest), factors(src))
    rank(dest) = rank(src)

    return dest
end

# Tensor error distributions
"""
    AbstractTensorErrorDistribution

Abstract type for tensor error distribution.
"""
abstract type AbstractTensorErrorDistribution end

"""
    WhiteNoise <: AbstractTensorErrorDistribution

White noise model of tensor errors `ε` with covariance matrix `Σ` of
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
        prod(size(ε)[1:end-1]) == size(Σ, 1) || throw(DimensionMismatch("number of elements in vec(εₜ) must equal rank of Σ."))

        return new{typeof(ε), Symmetric}(ε, Symmetric(Σ))
    end
end

"""
    TensorNormal <: AbstractTensorErrorDistribution

Tensor normal model of tensor errors `ε` with separable covariance matrix
`Σ` along each mode of the tensor.
"""
mutable struct TensorNormal{
    Tε<:AbstractArray,
    TΣ<:AbstractVector
} <: AbstractTensorErrorDistribution
    ε::Tε
    Σ::TΣ
    function TensorNormal(ε::AbstractArray, Σ::AbstractVector)
        length(Σ) == ndims(ε) - 1 || error("number of matrices in Σ must equal number of modes of εₜ.")
        all(size.(Σ, 1) .== size(ε)[1:end-1]) || throw(DimensionMismatch("dimensions of matrices in Σ must equal dimensions of modes of εₜ."))
        all(issymmetric, Σ) || error("all matrices in Σ must be symmetric.")

        return new{typeof(ε), typeof(Σ)}(ε, Symmetric.(Σ))
    end
end

# methods
resid(ε::AbstractTensorErrorDistribution) = ε.ε
Statistics.cov(ε::AbstractTensorErrorDistribution) = ε.Σ
Base.similar(ε::WhiteNoise) = WhiteNoise(similar(resid(ε)), similar(cov(ε)))
Base.similar(ε::TensorNormal) = TensorNormal(similar(resid(ε)), similar.(cov(ε)))
function Base.copyto!(dest::WhiteNoise, src::WhiteNoise)
    copyto!(resid(dest), resid(src))
    copyto!(cov(dest), cov(src))

    return dest
end
function Base.copyto!(dest::TensorNormal, src::TensorNormal)
    copyto!(resid(dest), resid(src))
    copyto!.(cov(dest), cov(src))

    return dest
end

# Tensor autoreggresive model
"""
    TensorAutoregression

Tensor autoregressive model with tensor error distribution `ε` and Kruskal tensor
representation `A`, potentially dynamic.
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
        size(y)[1:end-1] == size(resid(ε))[1:end-1] || throw(DimensionMismatch("dimensions of y and residuals must be equal."))
        all(size(y)[1:end-1] .== size.(factors(A), 1)) || throw(DimensionMismatch("dimensions of loadings must equal number of columns of y."))

        return new{typeof(y), typeof(ε), typeof(A)}(y, ε, A)
    end
end

# methods
data(model::TensorAutoregression) = model.y
coef(model::TensorAutoregression) = model.A
dist(model::TensorAutoregression) = model.ε
resid(model::TensorAutoregression) = resid(dist(model))
Statistics.cov(model::TensorAutoregression) = cov(dist(model))
factors(model::TensorAutoregression) = factors(coef(model))
loadings(model::TensorAutoregression) = loadings(coef(model))
rank(model::TensorAutoregression) = rank(coef(model))
Base.similar(model::TensorAutoregression) = TensorAutoregression(similar(data(model)), similar(dist(model)), similar(coef(model)))
function Base.copyto!(dest::TensorAutoregression, src::TensorAutoregression)
    copyto!(data(dest), data(src))
    copyto!(dist(dest), dist(src))
    copyto!(coef(dest), coef(src))

    return dest
end
Base.copy(model::TensorAutoregression) = copyto!(similar(model), model)