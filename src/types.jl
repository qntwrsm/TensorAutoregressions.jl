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
struct StaticKruskal{
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
struct DynamicKruskal{
    Tλ<:AbstractMatrix,
    Tα<:AbstractVector,
    Tϕ<:Diagonal,
    TΣ<:Diagonal,
    TU<:AbstractVector
} <: AbstractKruskal
    λ::Tλ
    α::Tα
    ϕ::Tϕ
    Σ::TΣ
    U::TU
    R::Int
    function DynamicKruskal(
        λ::AbstractMatrix,
        α::AbstractVector,
        ϕ::Diagonal, 
        Σ::Diagonal, 
        U::AbstractVector, 
        R::Integer
    )
        length(α) == R || error("α must be a vector of length R.")
        size(ϕ, 1) == R || error("ϕ must be a diagonal matrix of rank R.")
        size(Σ, 1) == R || error("Σ must be a diagonal matrix of rank R.")
        length(unique(size.(U, 2))) == 1 || throw(DimensionMismatch("all factor matrices must have the same number of columns."))
        size(λ, 1) == size(U[1], 2) == R || throw(DimensionMismatch("number of loadings and number of columns of factor matrices must equal rank R."))

        return new{typeof(λ), typeof(α), typeof(ϕ), typeof(Σ), typeof(U)}(λ, α, ϕ, Σ, U, R)
    end
end

# methods
factors(A::AbstractKruskal) = A.U
loadings(A::AbstractKruskal) = A.λ
rank(A::AbstractKruskal) = A.R
Base.size(A::AbstractKruskal) = tuple(size.(factors(A), 1)...)
Base.size(A::DynamicKruskal) = tuple(size.(factors(A), 1)..., size(loadings(A), 2))
full(A::AbstractKruskal) = tucker(loadings(A) .* I(length(factors(A)), rank(A)), factors(A))
full(A::DynamicKruskal) = tucker(I(length(factors(A)), rank(A)), factors(A))
intercept(A::DynamicKruskal) = A.α
dynamics(A::DynamicKruskal) = A.ϕ
cov(A::DynamicKruskal) = A.Σ
dof(A::StaticKruskal) = rank(A) + sum(length, factors(A))
dof(A::DynamicKruskal) = 3 * rank(A) + sum(length, factors(A))
Base.similar(A::StaticKruskal) = StaticKruskal(similar(loadings(A)), similar.(factors(A)), rank(A))
Base.similar(A::DynamicKruskal) = DynamicKruskal(similar(loadings(A)), similar(intercept(A)), similar(dynamics(A)), similar(cov(A)), similar.(factors(A)), rank(A))
function Base.copyto!(dest::StaticKruskal, src::StaticKruskal)
    copyto!(loadings(dest), loadings(src))
    copyto!.(factors(dest), factors(src))
    rank(dest) = rank(src)

    return dest
end
function Base.copyto!(dest::DynamicKruskal, src::DynamicKruskal)
    copyto!(loadings(dest), loadings(src))
    copyto!(intercept(dest), intercept(src))
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
struct WhiteNoise{
    Tε<:AbstractArray,
    TΣ<:Symmetric
} <: AbstractTensorErrorDistribution
    ε::Tε
    Σ::Symmetric
    function WhiteNoise(ε::AbstractArray, Σ::Symmetric)
        prod(size(ε)[1:end-1]) == size(Σ, 1) || throw(DimensionMismatch("number of elements in vec(εₜ) must equal rank of Σ."))

        return new{typeof(ε), typeof(Σ)}(ε, Σ)
    end
end

"""
    TensorNormal <: AbstractTensorErrorDistribution

Tensor normal model of tensor errors `ε` with separable covariance matrix
`Σ` along each mode of the tensor.
"""
struct TensorNormal{
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
cov(ε::AbstractTensorErrorDistribution; full::Bool=false) = ε.Σ
function cov(ε::TensorNormal; full::Bool=false)
    if full
        Σ = cov(ε)[end]
        for Σi ∈ reverse(cov(ε)[1:end-1])
            Σ = kron(Σ, Σi)
        end

        return Σ
    else
        return ε.Σ
    end
end
dof(ε::AbstractTensorErrorDistribution) = (sum(length, cov(ε)) + sum(size.(cov(ε), 1))) / 2
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
    AbstractTensorAutoregression <: StatisticalModel

Abstract type for tensor autoregressive model.
"""
abstract type AbstractTensorAutoregression <: StatisticalModel end

"""
    StaticTensorAutoregression <: AbstractTensorAutoregression

Tensor autoregressive model with tensor error distribution `ε` and static
Kruskal tensor representation `A`.
"""
struct StaticTensorAutoregression{
    Ty<:AbstractArray, 
    Tε<:AbstractTensorErrorDistribution,
    TA<:StaticKruskal,
    Tfixed<:NamedTuple
} <: AbstractTensorAutoregression
    y::Ty
    ε::Tε
    A::TA
    fixed::Tfixed
    function StaticTensorAutoregression(
        y::AbstractArray, 
        ε::AbstractTensorErrorDistribution, 
        A::StaticKruskal, 
        fixed::NamedTuple
    )
        dims = size(y)
        n = ndims(y) - 1
        size(y)[1:n] == size(resid(ε))[1:n] || throw(DimensionMismatch("dimensions of y and residuals must be equal."))
        all((dims[1:n]..., dims[1:n]...) .== size.(factors(A), 1)) || throw(DimensionMismatch("dimensions of loadings must equal number of columns of y."))

        return new{typeof(y), typeof(ε), typeof(A), typeof(fixed)}(y, ε, A, fixed)
    end
end

"""
    DynamicTensorAutoregression <: AbstractTensorAutoregression

Tensor autoregressive model with tensor error distribution `ε` and dynamic
Kruskal tensor representation `A`.
"""
struct DynamicTensorAutoregression{
    Ty<:AbstractArray, 
    Tε<:AbstractTensorErrorDistribution,
    TA<:DynamicKruskal,
    Tfixed<:NamedTuple
} <: AbstractTensorAutoregression
    y::Ty
    ε::Tε
    A::TA
    fixed::Tfixed
    function DynamicTensorAutoregression(
        y::AbstractArray, 
        ε::AbstractTensorErrorDistribution, 
        A::DynamicKruskal, 
        fixed::NamedTuple
    )
        dims = size(y)
        n = ndims(y) - 1
        size(y)[1:n] == size(resid(ε))[1:n] || throw(DimensionMismatch("dimensions of y and residuals must be equal."))
        all((dims[1:n]..., dims[1:n]...) .== size.(factors(A), 1)) || throw(DimensionMismatch("dimensions of loadings must equal number of columns of y."))
        ε isa WhiteNoise && throw(ArgumentError("dynamic model with white noise error not supported."))

        return new{typeof(y), typeof(ε), typeof(A), typeof(fixed)}(y, ε, A, fixed)
    end
end

# methods
data(model::AbstractTensorAutoregression) = model.y
coef(model::AbstractTensorAutoregression) = model.A
dist(model::AbstractTensorAutoregression) = model.ε
fixed(model::AbstractTensorAutoregression) = model.fixed 
resid(model::AbstractTensorAutoregression) = resid(dist(model))
cov(model::AbstractTensorAutoregression; full::Bool=false) = cov(dist(model), full=full)
factors(model::AbstractTensorAutoregression) = factors(coef(model))
loadings(model::AbstractTensorAutoregression) = loadings(coef(model))
rank(model::AbstractTensorAutoregression) = rank(coef(model))
nobs(model::AbstractTensorAutoregression) = last(size(data(model)))
dof(model::AbstractTensorAutoregression) = dof(coef(model)) + dof(dist(model))
Base.similar(model::StaticTensorAutoregression) = StaticTensorAutoregression(similar(data(model)), similar(dist(model)), similar(coef(model)), NamedTuple())
Base.similar(model::DynamicTensorAutoregression) = DynamicTensorAutoregression(similar(data(model)), similar(dist(model)), similar(coef(model)), NamedTuple())
function Base.copyto!(dest::AbstractTensorAutoregression, src::AbstractTensorAutoregression)
    copyto!(data(dest), data(src))
    copyto!(dist(dest), dist(src))
    copyto!(coef(dest), coef(src))
    fixed(dest) = fixed(src)

    return dest
end
Base.copy(model::AbstractTensorAutoregression) = copyto!(similar(model), model)

# Impulse response functions
"""
    AbstractIRF

Abstract type for impulse response functions (IRFs).
"""
abstract type AbstractIRF end

"""
    StaticIRF <: AbstractIRF

Static impulse response function (IRF) with `Ψ` as the IRF matrix, `lower` and
`upper` as the lower and upper bounds of the ``α``% confidence interval, and
`orth` as whether the IRF is orthogonalized.
"""
struct StaticIRF{TΨ<:AbstractArray} <: AbstractIRF
    Ψ::TΨ
    lower::TΨ
    upper::TΨ
    orth::Bool
    function StaticIRF(Ψ::AbstractArray, lower::AbstractArray, upper::AbstractArray, orth::Bool)
        return new{typeof(Ψ)}(Ψ, lower, upper, orth)
    end
end

"""
    DynamicIRF <: AbstractIRF

Dynamic impulse response function (IRF) with `Ψ` as the IRF matrix, `lower` and
`upper` as the lower and upper bounds of the ``α``% confidence interval, and
`orth` as whether the IRF is orthogonalized.
"""
struct DynamicIRF{TΨ<:AbstractArray} <: AbstractIRF
    Ψ::TΨ
    lower::TΨ
    upper::TΨ
    orth::Bool
    function DynamicIRF(Ψ::AbstractArray, lower::AbstractArray, upper::AbstractArray, orth::Bool)
        return new{typeof(Ψ)}(Ψ, lower, upper, orth)
    end
end

# methods
irf(irfs::AbstractIRF) = irfs.Ψ
irf(irfs::StaticIRF, impulse, response) = @view irf(irfs)[impulse..., response..., :]
irf(irfs::DynamicIRF, impulse, response) = @view irf(irfs)[impulse..., response..., :, :]
irf(irfs::DynamicIRF, impulse, response, time) = @view irf(irfs)[impulse..., response..., :, time]
lower(irfs::AbstractIRF) = irfs.lower
lower(irfs::StaticIRF, impulse, response) = @view lower(irfs)[impulse..., response..., :]
lower(irfs::DynamicIRF, impulse, response) = @view lower(irfs)[impulse..., response..., :, :]
lower(irfs::DynamicIRF, impulse, response, time) = @view lower(irfs)[impulse..., response..., :, time]
upper(irfs::AbstractIRF) = irfs.upper
upper(irfs::StaticIRF, impulse, response) = @view upper(irfs)[impulse..., response..., :]
upper(irfs::DynamicIRF, impulse, response) = @view upper(irfs)[impulse..., response..., :, :]
upper(irfs::DynamicIRF, impulse, response, time) = @view upper(irfs)[impulse..., response..., :, time]
orth(irfs::AbstractIRF) = irfs.orth 