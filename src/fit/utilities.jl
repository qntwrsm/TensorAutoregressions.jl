#=
utilities.jl

    Provides a collection of utility tools for fitting tensor autoregressive 
    models, such as initialization, convergence checks, and log-likelihood 
    evaluation.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/23/03
=#

"""
    objective(model) -> f

Wrapper for objective function evaluation for tensor autoregressive model
`model`.
"""
objective(model::AbstractTensorAutoregression) = loglike(model)
function objective(model::StaticTensorAutoregression)
    if dist(model) isa WhiteNoise
        return sse(model)
    else
        return loglike(model)
    end
end

"""
    sse(model) -> sse

Evaluate sum of squared errors of the tensor autoregressive model `model`.
"""
function sse(model::AbstractTensorAutoregression)
    update_resid!(model)
    return norm(resid(model))^2
end

"""
    loglike(model) -> ll

Evaluate log-likelihood of the tensor autoregressive model `model`.
"""
function loglike(model::StaticTensorAutoregression)
    dist(model) isa WhiteNoise && error("Log-likelihood not available for white noise error distribution.")

    dims = size(data(model))
    n = ndims(data(model)) - 1

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(model)))
    # inverse of Cholesky decompositions
    Cinv = inv.(getproperty.(C, :L))

    # log-likelihood
    # constant
    ll = -0.5 * (last(dims) - 1) * prod(dims[1:n]) * log(2π)
    # log determinant component
    for k = 1:n
        m = setdiff(1:n, k)
        ll -= 0.5 * (last(dims) - 1) * prod(dims[m]) * logdet(C[k])
    end
    # fit component
    update_resid!(model)
    ll -= 0.5 * norm(tucker(resid(model), Cinv))^2

    return ll
end

function loglike(model::DynamicTensorAutoregression)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # filter
    (_, _, v, F, _) = filter(model)

    # annihilator matrix
    (A_low, Z_basis) = collapse(model)
    M = Ref(I) .- Z_basis .* ((A_low .* Z_basis) .\ A_low)

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(model)))
    # inverse of Cholesky decompositions
    Cinv = inv.(getproperty.(C, :L))
    # low dimensional state-covariance matrix
    H_low = A_low .* Z_basis
    
    # dependent variable
    Z = tucker(selectdim(data(model), n+1, 2:last(dims)), Cinv)

    # log-likelihood
    # constant
    ll = -0.5 * (last(dims) - 1) * prod(dims[1:n]) * log(2π)
    for t = 1:last(dims)-1
        # filter component
        ll -= 0.5 * (logdet(F[t]) + dot(v[t], inv(F[t]), v[t]))
        # collapsed component
        et = M[t] * vec(selectdim(Z, n+1, t))
        ll -= 0.5 * norm(et)^2
        # projection component
        ll += 0.5 * (last(dims) - 1) * logdet(H_low[t])
    end
    # projection component
    for k = 1:n
        m = setdiff(1:n, k)
        ll -= 0.5 * (last(dims) - 1) * prod(dims[m]) * logdet(C[k])
    end

    return ll
end

"""
    update_resid!(model)

In-place update of the residuals of the tensor autoregressive model `model`.
"""
function update_resid!(model::StaticTensorAutoregression)
    dims = size(y)
    n = ndims(y) - 1

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

    # outer product of Kruskal factors
    U = [[factors(model)[i+n][:,r] * factors(model)[i][:,r]' for i = 1:n] for r = 1:rank(model)]

    # update residuals
    resid(model) .= y_lead
    for r in 1:rank(model)
        resid(model) .-= loadings(model)[r] .* tucker(y_lag, U[r])
    end

    return nothing
end

function update_resid!(model::DynamicTensorAutoregression)
    dims = size(y)
    n = ndims(y) - 1

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

    # outer product of Kruskal factors
    U = [[factors(model)[i+n][:,r] * factors(model)[i][:,r]' for i = 1:n] for r = 1:rank(model)]

    # regressor tensors
    X = [tucker(y_lag, U[r]) for r = 1:rank(model)]

    # update residuals
    resid(model) .= y_lead
    for (t, εt) in pairs(eachslice(resid(model), dims=n+1))
        for r = 1:rank(model)
            εt .-= loadings(model)[r,t] .* selectdim(X[r], n+1, t)
        end
    end

    return nothing
end

"""
    init!(model, method)

Initialize the tensor autoregressive model `model` by `method`, excluding the
fixed parameters indicated.

When `method` is set to `:data`: 
- Initialization of the Kruskal coefficient tensor is based on ridge regression of
the vectorized model combined with a CP decomposition. In case of a dynamic
Kruskal tensor the dynamic paramaters are obtained from the factor model
representation of the model.
- Initiliazation of the tensor error distribution is based on the sample
covariance estimate of the residuals. In case of separability of the covariance
matrix a mode specific sample covariance is calculated.

When `method` is set to `:random`:
- Initialization of the Kruskal coefficient tensor is based on a randomly sampled
CP decomposition. In case of a dynamic Kruskal tensor the dynamic paramaters are
obtained from the factor model representation of the model.
- Initiliazation of the tensor error distribution is based on a randomly sampled
covariance matrix from an inverse Wishart distribution.

When `method` is set to `:none` no initialization is performed and model is
assumed to have been initialized manually before fitting.
"""
function init!(model::AbstractTensorAutoregression, method::NamedTuple)
    # initialize Kruskal coefficient tensor
    if method.coef != :none
        init!(
            coef(model), 
            data(model), 
            get(fixed(model), :coef, NamedTuple()), 
            method.coef
        )
    end

    # initialize tensor error distribution
    if method.dist != :none
        init!(
            dist(model), 
            data(model), 
            coef(model), 
            get(fixed(model), :dist, NamedTuple()), 
            method.dist
        )
    end

    return nothing
end

"""
    init!(A, y, fixed, method)

Initialize the Kruskal coefficient tensor `A` given the data `y` using `method`,
excluding the fixed parameters indicated by `fixed`.

When `method` is set to `:data`: 
Initialization of the Kruskal coefficient tensor is based on ridge regression of
the vectorized model combined with a CP decomposition. 

When `method` is set to `:random`:
Initialization of the Kruskal coefficient tensor is based on a randomly sampled
CP decomposition.

In case of a dynamic Kruskal tensor the dynamic paramaters are obtained from the
factor model representation of the model.
"""
function init!(A::AbstractKruskal, y::AbstractArray, fixed::NamedTuple, method::Symbol)
    dims = size(y)
    n = ndims(y) - 1

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

    # dependent variable and regressor
    z = reshape(y_lead, :, last(dims)-1)
    x = reshape(y_lag, :, last(dims)-1)

    if method == :data
        # ridge regression
        F = hessenberg(x * x')
        M = z * x'
        # gridsearch
        γ = exp10.(range(0, 4, length=100))
        β = similar(γ, typeof(M))
        bic = similar(γ)
        for (i, γi) ∈ pairs(γ)
            β[i] = M / (F + γi * I)
            rss = norm(z - β[i] * x)^2
            df = tr(x' / (F + γi * I) * x)
            bic[i] = (last(dims) - 1) * log(rss * inv(last(dims) - 1)) + df * log(last(dims) - 1)
        end
        # optimal
        β_star = β[argmin(bic)]

        # CP decomposition
        cp = cp_als(tensorize(β_star, n+1:2n, (dims[1:n]..., dims[1:n]...)), rank(A))
    end
    # factors
    if haskey(fixed, :factors)
        factors(A) .= fixed.factors
    else
        if method == :data
            factors(A) .= cp.fmat
        elseif method == :random
            for k = 1:n, r = 1:rank(A)
                factors(A)[k][:,r] .= randn(dims[k])
                factors(A)[k][:,r] .*= inv(norm(factors(A)[k][:,r]))
                factors(A)[k+n][:,r] .= randn(dims[k])
                factors(A)[k+n][:,r] .*= inv(norm(factors(A)[k+n][:,r]))
            end
        end
    end
    # loadings
    if A isa StaticKruskal
        if haskey(fixed, :loadings)
            loadings(A) .= fixed.loadings
        else
            if method == :data
                loadings(A) .= cp.lambda
            elseif method == :random
                loadings(A) .= rand(rank(A))
            end
        end
    else
        xt = similar(x, size(x, 1), rank(A))
        for t = 1:last(dims)-1
            # select time series
            yt = selectdim(y, n+1, t)
            # regressors
            for r = 1:rank(A)
                # outer product of Kruskal factors
                U = [factors(A)[i+n][:,r] * factors(A)[i][:,r]' for i = 1:n]
                xt[:,r] = vec(tucker(yt, U))
            end
            loadings(A)[:,t] = xt \ z[:,t]
        end

        # dynamics
        λ_lead = @view loadings(A)[:,2:end]
        λ_lag = @view loadings(A)[:,1:end-1]
        β = hcat.(Ref(ones(last(dims) - 2)), eachrow(λ_lag)) .\ eachrow(λ_lead)
        for (r, βr) in enumerate(β)
            intercept(A)[r] = haskey(fixed, :intercept) ? fixed.intercept[r] : βr[1]
            dynamics(A).diag[r] .= haskey(fixed, :dynamics) ? fixed.dynamics.diag[r] : βr[2]
        end
        cov(A).data .= I - dynamics(A) * dynamics(A)'
    end

    return nothing
end

"""
    init!(ε, y, A, fixed, method)

Initialize the tensor error distribution `ε` given the data `y` and the Kruskal
coefficent tensor `A` using `method`, excluding the fixed parameters indicated
by `fixed`.

When `method` is set to `:data`:
Initiliazation of the tensor error distribution is based on the sample
covariance estimate of the residuals. In case of separability of the covariance
matrix a mode specific sample covariance is calculated.

When `method` is set to `:random`:
Initialization of the tensor error distribution is based on a randomly sampled
covariance matrix from an inverse Wishart distribution.
"""
function init!(
    ε::AbstractTensorErrorDistribution, 
    y::AbstractArray, 
    A::AbstractKruskal, 
    fixed::NamedTuple,
    method::Symbol
)
    dims = size(y)
    n = ndims(y) - 1

    # lag and lead variables
    y_lead = selectdim(y, n+1, 2:last(dims))
    y_lag = selectdim(y, n+1, 1:last(dims)-1)

    # error distribution
    resid(ε) .= y_lead
    for r = 1:rank(A)
        # outer product of Kruskal factors
        U = [factors(A)[i+n][:,r] * factors(A)[i][:,r]' for i = 1:n]
        if A isa StaticKruskal
            resid(ε) .-= loadings(A)[r] .* tucker(y_lag, U)
        else
            resid(ε) .-= reshape(loadings(A), ones(Int, n)..., :) .* tucker(y_lag, U)
        end
    end
    # covariance
    if ε isa WhiteNoise
        if haskey(fixed, :cov)
            cov(ε) .= fixed.cov
        else
            if method == :data
                cov(ε).data .= cov(reshape(resid(ε), :, 1:last(dims)-1), dims=2)
            elseif method == :random
                p = prod(dims[1:n])
                cov(ε).data .= rand(InverseWishart(p+2, I(p)))
            end
        end
    else
        scale = one(eltype(resid(ε)))
        for k = 1:n
            if haskey(fixed, :cov)
                cov(ε)[k] .= fixed.cov[k]
            else
                if method == :data
                    cov(ε)[k].data .= cov(matricize(resid(ε), k), dims=2)
                elseif method == :random
                    cov(ε)[k].data .= rand(InverseWishart(dims[k] + 2, I(dims[k])))
                end
            end
            if k < n
                scale *= norm(cov(ε)[k])
                lmul!(inv(norm(cov(ε)[k])), cov(ε)[k].data)
            else
                lmul!(scale, cov(ε)[k].data)
            end
        end
    end
    
    return nothing
end

"""
    absdiff(x, y) -> δ

Calculate the maximum absolute difference between `x` and `y`.
"""
absdiff(x::AbstractArray, y::AbstractArray) = mapreduce((xi, yi) -> abs(xi - yi), max, x, y)
absdiff(ε::WhiteNoise, ε_prev::WhiteNoise) = absdiff(cov(ε), cov(ε_prev))
absdiff(ε::TensorNormal, ε_prev::TensorNormal) = maximum(absdiff.(cov(ε), cov(ε_prev)))
function absdiff(A::StaticKruskal, A_prev::StaticKruskal)
    δ_loadings = absdiff(loadings(A), loadings(A_prev))
    δ_factors = maximum(absdiff.(factors(A), factors(A_prev)))

    return max(δ_loadings, δ_factors)
end
function absdiff(A::DynamicKruskal, A_prev::DynamicKruskal)
    δ_factors = maximum(absdiff.(factors(A), factors(A_prev)))
    δ_dynamics = absdiff(dynamics(A), dynamics(A_prev))
    δ_cov = absdiff(cov(A), cov(A_prev))

    return max(δ_factors, δ_dynamics, δ_cov)
end
function absdiff(model::AbstractTensorAutoregression, model_prev::AbstractTensorAutoregression)
    δ_coef = absdiff(coef(model), coef(model_prev))
    δ_dist = absdiff(dist(model), dist(model_prev))

    return max(δ_coef, δ_dist)
end