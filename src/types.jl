#=
types.jl

    Provides a collection of types for working with tensor autoregressive models, such as
    the Kruskal tensor for the autoregressive coefficients and tensor error distributions,
    as well as the main tensor autoregressive model itself.

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
struct StaticKruskal{Tλ <: AbstractVector, TU <: AbstractVector} <: AbstractKruskal
    λ::Tλ
    U::TU
    R::Int
    function StaticKruskal(λ::AbstractVector, U::AbstractVector, R::Integer)
        length(unique(size.(U, 2))) == 1 ||
            throw(DimensionMismatch("all factor matrices must have the same number of columns."))
        length(λ) == size(U[1], 2) == R ||
            throw(DimensionMismatch("number of loadings and number of columns of factor matrices must equal rank R."))

        return new{typeof(λ), typeof(U)}(λ, U, R)
    end
end

"""
    DynamicKruskal <: AbstractKruskal

Dynamic Kruskal tensor of rank `R` with dynamic loadings `λ` and factor matrices
`U`.
"""
struct DynamicKruskal{Tλ <: AbstractMatrix, Tα <: AbstractVector, Tϕ <: Diagonal,
                      TΣ <: Diagonal, TU <: AbstractVector} <: AbstractKruskal
    λ::Tλ
    α::Tα
    ϕ::Tϕ
    Σ::TΣ
    U::TU
    R::Int
    function DynamicKruskal(λ::AbstractMatrix, α::AbstractVector, ϕ::Diagonal, Σ::Diagonal,
                            U::AbstractVector, R::Integer)
        length(α) == R || error("α must be a vector of length R.")
        size(ϕ, 1) == R || error("ϕ must be a diagonal matrix of rank R.")
        size(Σ, 1) == R || error("Σ must be a diagonal matrix of rank R.")
        length(unique(size.(U, 2))) == 1 ||
            throw(DimensionMismatch("all factor matrices must have the same number of columns."))
        size(λ, 1) == size(U[1], 2) == R ||
            throw(DimensionMismatch("number of loadings and number of columns of factor matrices must equal rank R."))

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
function outer(A::AbstractKruskal)
    [[factors(A)[i + n][:, r] * factors(A)[i][:, r]' for i in 1:n] for r in 1:rank(A)]
end
intercept(A::DynamicKruskal) = A.α
dynamics(A::DynamicKruskal) = A.ϕ
cov(A::DynamicKruskal) = A.Σ
dof(A::StaticKruskal) = rank(A) + sum(length, factors(A))
dof(A::DynamicKruskal) = 3 * rank(A) + sum(length, factors(A))

# Tensor error distributions
"""
    AbstractTensorErrorDistribution

Abstract type for tensor error distribution.
"""
abstract type AbstractTensorErrorDistribution end

"""
    WhiteNoise <: AbstractTensorErrorDistribution

White noise model of tensor errors with covariance matrix `Σ` of ``vec(ε)``.
"""
struct WhiteNoise{TΣ <: Symmetric} <: AbstractTensorErrorDistribution
    Σ::Symmetric
end

"""
    TensorNormal <: AbstractTensorErrorDistribution

Tensor normal model of tensor errors `ε` with separable covariance matrix `Σ` along each
mode of the tensor.
"""
struct TensorNormal{TΣ <: AbstractVector} <: AbstractTensorErrorDistribution
    Σ::TΣ
    function TensorNormal(Σ::AbstractVector)
        all(issymmetric, Σ) || error("all matrices in Σ must be symmetric.")
        Σsym = Symmetric.(Σ)

        return new{typeof(Σsym)}(Σsym)
    end
end

# methods
cov(ε::AbstractTensorErrorDistribution; full::Bool = false) = ε.Σ
function cov(ε::TensorNormal; full::Bool = false)
    if full
        Σ = cov(ε)[end]
        for Σi in reverse(cov(ε)[1:(end - 1)])
            Σ = kron(Σ, Σi)
        end

        return Σ
    else
        return ε.Σ
    end
end
dof(ε::AbstractTensorErrorDistribution) = (sum(length, cov(ε)) + sum(size.(cov(ε), 1))) / 2

# Tensor autoreggresive model
"""
    AbstractTensorAutoregression <: StatisticalModel

Abstract type for tensor autoregressive model.
"""
abstract type AbstractTensorAutoregression <: StatisticalModel end

"""
    StaticTensorAutoregression <: AbstractTensorAutoregression

Tensor autoregressive model with tensor error distribution `ε` and static Kruskal tensor
representations `A`.
"""
struct StaticTensorAutoregression{Ty <: AbstractArray,
                                  Tε <: AbstractTensorErrorDistribution,
                                  TA <: AbstractVector} <: AbstractTensorAutoregression
    y::Ty
    ε::Tε
    A::TA
    function StaticTensorAutoregression(y::AbstractArray,
                                        ε::AbstractTensorErrorDistribution,
                                        A::AbstractVector)
        dims = size(y)
        n = ndims(y) - 1
        if ε isa WhiteNoise
            prod(dims[1:n]) == size(cov(ε), 1) ||
                throw(DimensionMismatch("dimensions of y and Σ must be equal."))
        elseif ε isa TensorNormal
            all(dims[1:n] .== size.(cov(ε), 1)) ||
                throw(DimensionMismatch("dimensions of y and Σ must be equal."))
        end
        for (p, Ap) in enumerate(A)
            Ap isa StaticKruskal ||
                throw(ArgumentError("only static Kruskal tensors allowed."))
            all((dims[1:n]..., dims[1:n]...) .== size.(factors(Ap), 1)) ||
                throw(DimensionMismatch("dimensions of loadings for lag" * str(p) *
                                        "must equal number of columns of y."))
        end

        return new{typeof(y), typeof(ε), typeof(A)}(y, ε, A)
    end
end

"""
    DynamicTensorAutoregression <: AbstractTensorAutoregression

Tensor autoregressive model with tensor error distribution `ε` and dynamic Kruskal tensor
representations `A`.
"""
struct DynamicTensorAutoregression{Ty <: AbstractArray,
                                   Tε <: AbstractTensorErrorDistribution,
                                   TA <: AbstractVector} <: AbstractTensorAutoregression
    y::Ty
    ε::Tε
    A::TA
    function DynamicTensorAutoregression(y::AbstractArray,
                                         ε::AbstractTensorErrorDistribution,
                                         A::AbstractVector)
        dims = size(y)
        n = ndims(y) - 1
        if ε isa WhiteNoise
            throw(ArgumentError("dynamic model with white noise error not supported."))
        elseif ε isa TensorNormal
            all(dims[1:n] .== size.(cov(ε), 1)) ||
                throw(DimensionMismatch("dimensions of y and Σ must be equal."))
        end
        for (p, Ap) in enumerate(A)
            Ap isa DynamicKruskal ||
                throw(ArgumentError("only dynamic Kruskal tensors allowed."))
            all((dims[1:n]..., dims[1:n]...) .== size.(factors(Ap), 1)) ||
                throw(DimensionMismatch("dimensions of loadings for lag" * str(p) *
                                        "must equal number of columns of y."))
        end

        return new{typeof(y), typeof(ε), typeof(A)}(y, ε, A)
    end
end

# methods
data(model::AbstractTensorAutoregression) = model.y
coef(model::AbstractTensorAutoregression) = model.A
dist(model::AbstractTensorAutoregression) = model.ε
cov(model::AbstractTensorAutoregression; full::Bool = false) = cov(dist(model), full = full)
factors(model::AbstractTensorAutoregression) = factors.(coef(model))
loadings(model::AbstractTensorAutoregression) = loadings.(coef(model))
rank(model::AbstractTensorAutoregression) = rank.(coef(model))
nobs(model::AbstractTensorAutoregression) = last(size(data(model)))
dof(model::AbstractTensorAutoregression) = sum(dof, coef(model)) + dof(dist(model))

# Impulse response functions
"""
    AbstractIRF

Abstract type for impulse response functions (IRFs).
"""
abstract type AbstractIRF end

"""
    StaticIRF <: AbstractIRF

Static impulse response function (IRF) with `Ψ` as the IRF matrix, `lower` and `upper` as
the lower and upper bounds of the ``α``% confidence interval, and `orth` as whether the IRF
is orthogonalized.
"""
struct StaticIRF{TΨ <: AbstractArray} <: AbstractIRF
    Ψ::TΨ
    lower::TΨ
    upper::TΨ
    orth::Bool
    function StaticIRF(Ψ::AbstractArray, lower::AbstractArray, upper::AbstractArray,
                       orth::Bool)
        return new{typeof(Ψ)}(Ψ, lower, upper, orth)
    end
end

"""
    DynamicIRF <: AbstractIRF

Dynamic impulse response function (IRF) with `Ψ` as the IRF matrix, `lower` and `upper` as
the lower and upper bounds of the ``α``% confidence interval, and `orth` as whether the IRF
is orthogonalized.
"""
struct DynamicIRF{TΨ <: AbstractArray} <: AbstractIRF
    Ψ::TΨ
    lower::TΨ
    upper::TΨ
    orth::Bool
    function DynamicIRF(Ψ::AbstractArray, lower::AbstractArray, upper::AbstractArray,
                        orth::Bool)
        return new{typeof(Ψ)}(Ψ, lower, upper, orth)
    end
end

# methods
irf(irfs::AbstractIRF) = irfs.Ψ
irf(irfs::StaticIRF, impulse, response) = @view irf(irfs)[impulse..., response..., :]
irf(irfs::DynamicIRF, impulse, response) = @view irf(irfs)[impulse..., response..., :, :]
function irf(irfs::DynamicIRF, impulse, response, time)
    @view irf(irfs)[impulse..., response..., :, time]
end
lower(irfs::AbstractIRF) = irfs.lower
lower(irfs::StaticIRF, impulse, response) = @view lower(irfs)[impulse..., response..., :]
function lower(irfs::DynamicIRF, impulse, response)
    @view lower(irfs)[impulse..., response..., :, :]
end
function lower(irfs::DynamicIRF, impulse, response, time)
    @view lower(irfs)[impulse..., response..., :, time]
end
upper(irfs::AbstractIRF) = irfs.upper
upper(irfs::StaticIRF, impulse, response) = @view upper(irfs)[impulse..., response..., :]
function upper(irfs::DynamicIRF, impulse, response)
    @view upper(irfs)[impulse..., response..., :, :]
end
function upper(irfs::DynamicIRF, impulse, response, time)
    @view upper(irfs)[impulse..., response..., :, time]
end
orth(irfs::AbstractIRF) = irfs.orth
