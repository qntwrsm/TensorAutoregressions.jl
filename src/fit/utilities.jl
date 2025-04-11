#=
utilities.jl

    Provides a collection of utility tools for fitting tensor autoregressive models, such as
    initialization, convergence checks, and log-likelihood evaluation.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/23/03
=#

"""
    objective(model) -> f

Wrapper for objective function evaluation for tensor autoregressive model `model`.
"""
objective(model::AbstractTensorAutoregression) = loglikelihood(model)
function objective(model::StaticTensorAutoregression)
    if dist(model) isa WhiteNoise
        return rss(model)
    else
        return loglikelihood(model)
    end
end

rss(model::AbstractTensorAutoregression) = norm(residuals(model))^2

function loglikelihood(model::StaticTensorAutoregression)
    dist(model) isa WhiteNoise &&
        error("Log-likelihood not available for white noise error distribution.")

    d = dims(model)
    n = length(d)

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(model)))
    # inverse of Cholesky decompositions
    Cinv = inv.(getproperty.(C, :L))

    # log-likelihood
    # constant
    ll = -0.5 * (nobs(model) - lags(model)) * prod(d) * log(2π)
    # log determinant component
    for k in 1:n
        k_ = setdiff(1:n, k)
        ll -= 0.5 * (nobs(model) - lags(model)) * prod(d[k_]) * logdet(C[k])
    end
    # fit component
    ll -= 0.5 * norm(tucker(residuals(model), Cinv))^2

    return ll
end
function loglikelihood(model::DynamicTensorAutoregression)
    d = dims(model)
    n = length(d)

    # concentration matrix
    Hinv = concentration(model, full = true)

    # collapsed state space system
    (y_low, Z_low, H_low, M) = collapse(model, objective = true)
    (c, T, Q) = state_transition_params(model)
    (a1, P1) = state_space_init(model)

    # filter
    (v, F) = _filter_likelihood(y_low, Z_low, H_low, c, T, Q, a1, P1)

    # log-likelihood
    # constant
    ll = -0.5 * (nobs(model) - lags(model)) * prod(d) * log(2π)
    for t in 1:(nobs(model) - lags(model))
        # filter component
        ll -= 0.5 * (logdet(F[t]) + dot(v[t], F[t] \ v[t]))
        # collapsed component
        et = M[t] * vec(selectdim(data(model), n + 1, t + lags(model)))
        ll -= 0.5 * dot(et, Hinv, et)
        # projection component
        ll += 0.5 * logdet(H_low[t])
    end
    # projection component
    for k in 1:n
        k_ = setdiff(1:n, k)
        ll -= 0.5 * (nobs(model) - lags(model)) * prod(d[k_]) * logdet(cov(model)[k])
    end

    return ll
end

"""
    residuals(model) -> ε

Return the residuals of the tensor autoregressive model `model`.
"""
function residuals(model::StaticTensorAutoregression)
    n = length(dims(model))

    # lag and lead variables
    y_lead = selectdim(data(model), n + 1, (lags(model) + 1):nobs(model))
    y_lags = [selectdim(data(model), n + 1, (lags(model) - p + 1):(nobs(model) - p))
              for p in 1:lags(model)]

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # update residuals
    ε = copy(y_lead)
    for (p, Rp) in pairs(rank(model))
        for r in 1:Rp
            ε .-= loadings(model)[p][r] .* tucker(y_lags[p], U[p][r])
        end
    end

    return ε
end
function residuals(model::DynamicTensorAutoregression)
    n = length(dims(model))

    # lag and lead variables
    y_lead = selectdim(data(model), n + 1, (lags(model) + 1):nobs(model))
    y_lags = [selectdim(data(model), n + 1, (lags(model) - p + 1):(nobs(model) - p))
              for p in 1:lags(model)]

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # regressor tensors
    X = [[tucker(y_lags[p], U[p][r]) for r in 1:Rp] for (p, Rp) in pairs(rank(model))]

    # update residuals
    ε = copy(y_lead)
    for (t, εt) in pairs(eachslice(ε, dims = n + 1))
        for (p, Rp) in pairs(rank(model))
            for r in 1:Rp
                εt .-= loadings(model)[p][r, t] .* selectdim(X[p][r], n + 1, t)
            end
        end
    end

    return ε
end

"""
    init!(model, method)

Initialize the tensor autoregressive model `model` by `method`.

When `method` is set to `:data`:

  - Initialization of the Kruskal coefficient tensor is based on ridge regression of the
    vectorized model combined with a CP decomposition. In case of a dynamic Kruskal tensor
    the dynamic paramaters are obtained from the factor model representation of the model.
  - Initiliazation of the tensor error distribution is based on the sample covariance
    estimate of the residuals. In case of separability of the covariance matrix a mode
    specific sample covariance is calculated.

When `method` is set to `:random`:

  - Initialization of the Kruskal coefficient tensor is based on a randomly sampled CP
    decomposition. In case of a dynamic Kruskal tensor the dynamic paramaters are obtained
    from the factor model representation of the model.
  - Initiliazation of the tensor error distribution is based on a randomly sampled
    covariance matrix from an inverse Wishart distribution.

When `method` is set to `:none` no initialization is performed and model is assumed to have
been initialized manually before fitting.
"""
function init!(model::AbstractTensorAutoregression, method::NamedTuple)
    # initialize Kruskal coefficient tensor
    method.coef != :none && init_kruskal!(model, method.coef)
    # initialize tensor error distribution
    method.dist != :none && init_dist!(model, method.dist)

    return nothing
end

"""
    init_kruskal!(model, method)

Initialize the Kruskal coefficient tensors of the tensor autoregressive model `model` using
`method`.

When `method` is set to `:data`:
Initialization of the Kruskal coefficient tensor is based on ridge regression of the
vectorized model combined with a CP decomposition.

When `method` is set to `:random`:
Initialization of the Kruskal coefficient tensor is based on a randomly sampled CP
decomposition.

In case of a dynamic Kruskal tensor the dynamic paramaters are obtained from the factor
model representation of the model.
"""
function init_kruskal!(model::AbstractTensorAutoregression, method::Symbol)
    d = dims(model)
    n = length(d)
    N = prod(d)
    kruskal_shape = (d..., d...)
    Rc = cumsum(rank(model))

    # lag and lead variables
    y_lead = selectdim(data(model), n + 1, (lags(model) + 1):nobs(model))
    y_lags = [selectdim(data(model), n + 1, (lags(model) - p + 1):(nobs(model) - p))
              for p in 1:lags(model)]

    # dependent variable and regressor
    z = reshape(y_lead, :, nobs(model) - lags(model))
    x = vcat(reshape.(y_lags, :, nobs(model) - lags(model))...)

    if method == :data
        # linear regression
        β = z / x
        B = [tensorize(β[:, ((p - 1) * N + 1):(p * N)], (n + 1):(2n), kruskal_shape)
             for p in 1:lags(model)]

        # CP decomposition
        Astatic = []
        for (p, Rp) in pairs(rank(model))
            # trials
            (opt_trial, opt_obj) = cp(B[p], Rp, tolerance = 1e-6)
            for _ in 1:100
                (trial, obj_trial) = cp(B[p], Rp, tolerance = 1e-6)
                if obj_trial > opt_obj
                    opt_trial = trial
                    opt_obj = obj_trial
                end
            end
            push!(Astatic, opt_trial)
        end
    end
    # factors
    if method == :data
        for (p, Ap) in pairs(coef(model))
            factors(Ap) .= factors(Astatic[p])
        end
    elseif method == :random
        for Ap in coef(model), k in 1:n, r in 1:rank(Ap)
            factors(Ap)[k][:, r] .= randn(d[k])
            factors(Ap)[k][:, r] .*= inv(norm(factors(Ap)[k][:, r]))
            factors(Ap)[k + n][:, r] .= randn(d[k])
            factors(Ap)[k + n][:, r] .*= inv(norm(factors(Ap)[k + n][:, r]))
        end
    end

    # loadings
    if all(x -> isa(x, StaticKruskal), coef(model))
        if method == :data
            for (p, Ap) in pairs(coef(model))
                loadings(Ap) .= loadings(Astatic[p])
            end
        elseif method == :random
            for Ap in coef(model)
                loadings(Ap) .= randn(rank(Ap))
            end
        end
    else
        # outer product of Kruskal factors
        L = loading_matrix(model)
        for (t, Lt) in pairs(L)
            λt = Lt \ view(z, :, t)
            for (j, λjt) in pairs(λt)
                p = sum(x -> isless(x, j), Rc) + 1
                r = j - get(Rc, p - 1, 0)
                loadings(model)[p][r, t] = λjt
            end
        end

        # transition dynamics
        for Ap in coef(model)
            for r in 1:rank(Ap)
                y = loadings(Ap)[r, 2:end]
                x = hcat(loadings(Ap)[r, 1:end-1], ones(length(y)))

                # linear regression
                β = x \ y
                resid = y - x * β

                dynamics(Ap).diag[r] = β[1]
                intercept(Ap)[r] = β[2]
                cov(Ap).diag[r] = var(resid)
            end
        end
    end

    return nothing
end

"""
    init_dist!(model, method)

Initialize the tensor error distribution of the tensor autoregressive model `model` using
`method`.

When `method` is set to `:data`:
Initiliazation of the tensor error distribution is based on the sample covariance estimate
of the residuals. In case of separability of the covariance matrix a mode specific sample
covariance is calculated.

When `method` is set to `:random`:
Initialization of the tensor error distribution is based on a randomly sampled covariance
matrix from an inverse Wishart distribution.
"""
function init_dist!(model::AbstractTensorAutoregression, method::Symbol)
    d = dims(model)
    n = length(d)

    # error distribution
    resid = residuals(model)
    # covariance
    if dist(model) isa WhiteNoise
        if method == :data
            cov(model).data .= cov(reshape(resid, :, 1:(nobs(model) - lags(model))),
                                   dims = 2)
        elseif method == :random
            N = prod(d)
            cov(model).data .= rand(InverseWishart(N + 2, I(N)))
        end
    else
        scale = one(eltype(resid))
        for k in 1:n
            if method == :data
                cov(model)[k].data .= cov(matricize(resid, k), dims = 2)
            elseif method == :random
                cov(model)[k].data .= rand(InverseWishart(d[k] + 2, I(d[k])))
            end
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
