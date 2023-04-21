#=
utilities.jl

    Provides a collection of utility tools for fitting tensor autoregressive 
    models, such as initialization, convergence checks, and log-likelihood 
    evaluation.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/23/03
=#

"""
    loglike(model) -> ll

Evaluate the log-likelihood of the tensor autoregressive model `model`.
"""
function loglike(model::TensorAutoregression)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # collapsed state space system
    (y_star, Z_star, a1, P1) = state_space(data(model), coef(model), dist(model))
    # filter
    (_, _, v, F, _) = filter(
        y_star, 
        Z_star,
        intercept(coef(model)),
        dynamics(coef(model)), 
        cov(coef(model)), 
        a1, 
        P1
    )

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(model)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]

    # outer product of Kruskal factors
    U = [factors(model)[i] * factors(model)[i+n]' for i = 1:n]

    # scaling
    S = [Cinv[i] * U[i] for i = 1:n]
    
    # dependent variable and regressor
    Z = tucker(selectdim(data(model), n+1, 2:last(dims)), Cinv, 1:n)
    X = tucker(selectdim(data(model), n+1, 1:last(dims)-1), S, 1:n)

    # log-likelihood
    # constant
    ll = -0.5 * (last(dims) - 1) * prod(dims[1:n]) * log(2π)
    for t = 1:last(dims)-1
        # filter component
        ll -= 0.5 * (logdet(F[t]) + dot(v[t], inv(F[t]), v[t]))
        # collapsed component
        et = selectdim(Z, n+1, t) - y_star[t] .* selectdim(X, n+1, t)
        ll -= 0.5 * norm(et)^2
    end
    # projection matrix component
    for k = 1:n
        m = setdiff(1:n, k)
        ll -= 0.5 * (last(dims) - 1) * prod(dims[m]) * logdet(C[k])
    end

    return ll
end

"""
    init!(model)

Initialize the tensor autoregressive model `model`.

Initialization of the Kruskal coefficient tensor is based on ridge regression of
the vectorized model combined with a CP decomposition. In case of a dynamic
Kruskal tensor the dynamic paramaters are obtained from the factor model
representation of the model.
Initiliazation of the tensor error distribution is based on the sample
covariance estimate of the residuals. In case of separability of the covariance
matrix a mode specific sample covariance is calculated.
"""
function init!(model::TensorAutoregression)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # lag and lead variables
    y_lead = selectdim(data(model), n+1, 2:last(dims))
    y_lag = selectdim(data(model), n+1, 1:last(dims)-1)

    # dependent variable and regressor
    z = reshape(y_lead, :, last(dims)-1)
    x = reshape(y_lag, :, last(dims)-1)

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
    cp = cp_als(reshape(β_star, dims[1:n]..., dims[1:n]...), rank(model))
    # factors
    factors(model) .= cp.fmat
    # loadings
    if coef(model) isa StaticKruskal
        loadings(model) .= sign.(cp.lambda) .* min.(abs.(cp.lambda), .9)
    else
        xt = similar(x, size(x, 1), rank(model))
        for t = 1:last(dims)-1
            # select time series
            yt = selectdim(data(model), n+1, t)
            # regressors
            for r = 1:rank(model)
                # outer product of Kruskal factors
                U = [factors(model)[i][:,r] * factors(model)[i+n][:,r]' for i = 1:n]
                xt[:,r] = vec(tucker(yt, U, 1:n))
            end
            loadings(model)[:,t] = xt \ z[:,t]
        end

        # dynamics
        λ_lead = @view loadings(model)[:,2:end]
        λ_lag = @view loadings(model)[:,1:end-1]
        β = λ_lead / vcat(ones(1,last(dims)-2), λ_lag)
        intercept(coef(model)) .= β[1]
        dynamics(coef(model)) .= β[2]
        cov(coef(model)).data .= I - dynamics(coef(model)) * dynamics(coef(model))'
    end
    
    # error distribution
    resid(model) .= y_lead
    for r = 1:rank(model)
        # outer product of Kruskal factors
        U = [factors(model)[i][:,r] * factors(model)[i+n][:,r]' for i = 1:n]
        if coef(model) isa StaticKruskal
            resid(model) .-= loadings(model)[r] .* tucker(y_lag, U, 1:n)
        else
            resid(model) .-= reshape(loadings(model), ones(Int, n)..., :) .* tucker(y_lag, U, 1:n)
        end
    end
    # covariance
    if dist(model) isa WhiteNoise
        cov(model).data .= cov(reshape(resid(model), :, 1:last(dims)-1), dims=2)
    else
        scale = one(eltype(resid(model)))
        for k = 1:n
            cov(model)[k].data .= cov(matricize(resid(model), k), dims=2)
            if k < n
                scale *= norm(cov(model)[k])
                lmul!(inv(norm(cov(model)[k])), cov(model)[k].data)
            else
                lmul!(scale, cov(model)[k].data)
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
function absdiff(model::TensorAutoregression, model_prev::TensorAutoregression)
    δ_coef = absdiff(coef(model), coef(model_prev))
    δ_dist = absdiff(dist(model), dist(model_prev))

    return max(δ_coef, δ_dist)
end