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
function full(A::DynamicKruskal)
    n = length(factors(A))
    U = [[factors(A)[i][:, r] for i in 1:n] for r in 1:rank(A)]

    return [tucker(ones((1 for _ in 1:n)...), U[r]) for r in 1:rank(A)]
end
function outer(A::AbstractKruskal)
    n = length(factors(A)) ÷ 2

    return [[factors(A)[i + n][:, r] * factors(A)[i][:, r]' for i in 1:n]
            for r in 1:rank(A)]
end
intercept(A::DynamicKruskal) = A.α
dynamics(A::DynamicKruskal) = A.ϕ
cov(A::DynamicKruskal) = A.Σ
dof(A::StaticKruskal) = rank(A) + sum(length, factors(A))
dof(A::DynamicKruskal) = 3 * rank(A) + sum(length, factors(A))
params(A::StaticKruskal) = vcat(vec.(factors(A))..., loadings(A))
function params(A::DynamicKruskal)
    vcat(vec.(factors(A))..., intercept(A), dynamics(A).diag, cov(A).diag)
end
function Base.copy(A::StaticKruskal)
    StaticKruskal(copy(loadings(A)), deepcopy(factors(A)), rank(A))
end
function Base.copy(A::DynamicKruskal)
    DynamicKruskal(copy(loadings(A)), copy(intercept(A)), copy(dynamics(A)), copy(cov(A)),
                   deepcopy(factors(A)), rank(A))
end
function fixsign!(A::StaticKruskal)
    T = eltype(loadings(A))
    n = length(factors(A))

    # fix signs of loadings
    idx = findall(λ -> λ < 0, loadings(A))
    loadings(A)[idx] .*= -one(T)
    factors(A)[1][:, idx] .*= -one(T)

    # fix signs of factors
    signs = zeros(T, n)
    for r in 1:rank(A)
        # find sign of largest factor elements
        for (k, Uk) in pairs(factors(A))
            idx = argmax(abs.(Uk[:, r]))
            signs[k] = sign(Uk[idx, r])
        end

        # flip signs
        negatives = findall(s -> s == -1, signs)
        for i in 1:(length(negatives) - length(negatives) % 2)
            k = negatives[i]
            factors(A)[k][:, r] .*= -one(T)
        end
    end

    return nothing
end
function fixsign!(A::DynamicKruskal)
    T = eltype(loadings(A))
    n = length(factors(A))

    # fix signs of factors
    signs = zeros(T, n)
    for r in 1:rank(A)
        # find sign of largest factor elements
        for (k, Uk) in pairs(factors(A))
            idx = argmax(abs.(Uk[:, r]))
            signs[k] = sign(Uk[idx, r])
        end

        # flip signs
        negatives = findall(s -> s == -1, signs)
        for i in 1:(length(negatives) - length(negatives) % 2)
            k = negatives[i]
            factors(A)[k][:, r] .*= -one(T)
        end
    end

    return nothing
end
function fixorder!(A::StaticKruskal)
    # permutation
    p = sortperm(loadings(A), rev = true)

    # factors
    for Uk in factors(A)
        Uk .= Uk[:, p]
    end
    # loadings
    loadings(A) .= loadings(A)[p]

    return nothing
end

function fixorder!(A::DynamicKruskal)
    # permutation
    p = sortperm(dynamics(A).diag, rev = true)

    # factors
    for Uk in factors(A)
        Uk .= Uk[:, p]
    end
    # loadings
    loadings(A) .= loadings(A)[p, :]
    # transition dynamics
    intercept(A) .= intercept(A)[p]
    dynamics(A).diag .= dynamics(A).diag[p]
    cov(A).diag .= cov(A).diag[p]

    return nothing
end

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
    Σ::TΣ
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
concentration(ε::AbstractTensorErrorDistribution; full::Bool = false) = inv(cov(ε))
function concentration(ε::TensorNormal; full::Bool = false)
    if full
        Ω = inv(cov(ε)[end])
        for Σi in reverse(cov(ε)[1:(end - 1)])
            Ω = kron(Ω, inv(Σi))
        end

        return Ω
    else
        return inv.(cov(ε))
    end
end
dof(ε::AbstractTensorErrorDistribution) = (sum(length, cov(ε)) + sum(size.(cov(ε), 1))) / 2
params(ε::WhiteNoise) = vec(cov(ε))
params(ε::TensorNormal) = vcat(vec.(cov(ε))...)
Base.copy(ε::WhiteNoise) = WhiteNoise(copy(cov(ε)))
Base.copy(ε::TensorNormal) = TensorNormal(deepcopy(cov(ε)))

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
        d = size(y)
        n = ndims(y) - 1
        if ε isa WhiteNoise
            prod(d[1:n]) == size(cov(ε), 1) ||
                throw(DimensionMismatch("dimensions of y and Σ must be equal."))
        elseif ε isa TensorNormal
            all(d[1:n] .== size.(cov(ε), 1)) ||
                throw(DimensionMismatch("dimensions of y and Σ must be equal."))
        end
        for (p, Ap) in enumerate(A)
            Ap isa StaticKruskal ||
                throw(ArgumentError("only static Kruskal tensors allowed."))
            all((d[1:n]..., d[1:n]...) .== size.(factors(Ap), 1)) ||
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
        d = size(y)
        n = ndims(y) - 1
        if ε isa WhiteNoise
            throw(ArgumentError("dynamic model with white noise error not supported."))
        elseif ε isa TensorNormal
            all(d[1:n] .== size.(cov(ε), 1)) ||
                throw(DimensionMismatch("dimensions of y and Σ must be equal."))
        end
        for (p, Ap) in enumerate(A)
            Ap isa DynamicKruskal ||
                throw(ArgumentError("only dynamic Kruskal tensors allowed."))
            all((d[1:n]..., d[1:n]...) .== size.(factors(Ap), 1)) ||
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
function concentration(model::AbstractTensorAutoregression; full::Bool = false)
    concentration(dist(model), full = full)
end
factors(model::AbstractTensorAutoregression) = factors.(coef(model))
loadings(model::AbstractTensorAutoregression) = loadings.(coef(model))
rank(model::AbstractTensorAutoregression) = rank.(coef(model))
lags(model::AbstractTensorAutoregression) = length(coef(model))
dims(model::AbstractTensorAutoregression) = Base.front(size(data(model)))
nobs(model::AbstractTensorAutoregression) = last(size(data(model)))
dof(model::AbstractTensorAutoregression) = sum(dof, coef(model)) + dof(dist(model))
function params(model::AbstractTensorAutoregression)
    vcat(params.(coef(model))..., params(dist(model)))
end
function Base.copy(model::StaticTensorAutoregression)
    StaticTensorAutoregression(copy(data(model)), copy(dist(model)), copy.(coef(model)))
end
function Base.copy(model::DynamicTensorAutoregression)
    DynamicTensorAutoregression(copy(data(model)), copy(dist(model)), copy.(coef(model)))
end

# Impulse response functions
"""
    AbstractIRF

Abstract type for (generalized) impulse response functions (IRFs).
"""
abstract type AbstractIRF end

"""
    StaticIRF <: AbstractIRF

Static (generalized) impulse response function (IRF) with `Ψ` as the IRF matrix, `lower` and
`upper` as the lower and upper bounds of the ``α``% confidence interval, and `orth` as
whether the IRF is orthogonalized.
"""
struct StaticIRF{TΨ <: AbstractArray} <: AbstractIRF
    Ψ::TΨ
    lower::TΨ
    upper::TΨ
    function StaticIRF(Ψ::AbstractArray, lower::AbstractArray, upper::AbstractArray)
        return new{typeof(Ψ)}(Ψ, lower, upper)
    end
end

"""
    DynamicIRF <: AbstractIRF

Dynamic (generalized) impulse response function (IRF) with `Ψ` as the IRF matrix, `lower`
and `upper` as the lower and upper bounds of the ``α``% confidence interval, and `orth` as
whether the IRF is orthogonalized.
"""
struct DynamicIRF{TΨ <: AbstractArray} <: AbstractIRF
    Ψ::TΨ
    lower::TΨ
    upper::TΨ
    function DynamicIRF(Ψ::AbstractArray, lower::AbstractArray, upper::AbstractArray)
        return new{typeof(Ψ)}(Ψ, lower, upper)
    end
end

# methods
irf(irfs::AbstractIRF) = irfs.Ψ
irf(irfs::StaticIRF, impulse, response) = @view irf(irfs)[impulse..., response..., :]
irf(irfs::DynamicIRF, response) = @view irf(irfs)[response..., :]
lower(irfs::AbstractIRF) = irfs.lower
lower(irfs::StaticIRF, impulse, response) = @view lower(irfs)[impulse..., response..., :]
lower(irfs::DynamicIRF, response) = @view lower(irfs)[response..., :]
upper(irfs::AbstractIRF) = irfs.upper
upper(irfs::StaticIRF, impulse, response) = @view upper(irfs)[impulse..., response..., :]
upper(irfs::DynamicIRF, response) = @view upper(irfs)[response..., :]
