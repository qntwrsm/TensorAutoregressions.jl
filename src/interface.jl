#=
interface.jl

    Provides a collection of interface tools for working with tensor autoregressive models,
    such as estimation, forecasting, and impulse response analysis.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/03/02
=#

"""
    TensorAutoregression(y, R; dynamic=false, dist=:white_noise) -> model

Constructs a tensor autoregressive model for data `y` with autoregressive coefficient
tensors of rank `R` for lags 1 until ``P``, potentially dynamic, and tensor error
distribution `dist`.
"""
function TensorAutoregression(y::AbstractArray, R::Dims; dynamic::Bool = false,
                              dist::Symbol = :white_noise)
    d = size(y)
    n = ndims(y) - 1
    p = length(R)

    # check model specification
    dynamic && dist == :white_noise &&
        throw(ArgumentError("dynamic model with white noise error not supported."))

    # instantiate Kruskal autoregressive tensors
    if dynamic
        A = [DynamicKruskal(similar(y, Rp, last(d) - p), similar(y, Rp),
                            Diagonal(similar(y, Rp)), Diagonal(similar(y, Rp)),
                            [similar(y, d[i - n * ((i - 1) ÷ n)], Rp) for i in 1:(2n)],
                            Rp) for Rp in R]
        model = DynamicTensorAutoregression
    else
        A = [StaticKruskal(similar(y, Rp),
                           [similar(y, d[i - n * ((i - 1) ÷ n)], Rp) for i in 1:(2n)],
                           Rp) for Rp in R]
        model = StaticTensorAutoregression
    end

    # instantiate tensor error distribution
    N = prod(d[1:n])
    if dist == :white_noise
        ε = WhiteNoise(Symmetric(similar(y, N, N)))
    elseif dist == :tensor_normal
        ε = TensorNormal([Symmetric(similar(y, d[i], d[i])) for i in 1:n])
    else
        throw(ArgumentError("distribution $dist not supported."))
    end

    return model(y, ε, A)
end

"""
    TensorAutoregression(dims, R; dynamic=false, dist=:white_noise) -> model

Constructs a tensor autoregressive model of dimensions `dims` with autoregressive
coefficient tensors of rank `R` for lags 1 until ``P``, potentially dynamic, and tensor
error distribution `dist`.
"""
function TensorAutoregression(dims::Dims, R::Dims; dynamic::Bool = false,
                              dist::Symbol = :white_noise)
    TensorAutoregression(zeros(dims), R, dynamic = dynamic, dist = dist)
end

"""
    simulate(model; burn=100, rng=Xoshiro()) -> sim

Simulate data from the tensor autoregressive model described by `model` and
return a new instance with the simulated data, using random number generator
`rng` and apply a burn-in period of `burn`.
"""
function simulate(model::StaticTensorAutoregression; burn::Integer = 100,
                  rng::AbstractRNG = Xoshiro())
    d = dims(model)
    n = length(d)

    # tensor error distribution
    ε = simulate(dist(model), nobs(model) + burn, rng)

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # burn-in
    y_burn = similar(data(model), d..., burn)
    for (t, yt) in pairs(eachslice(y_burn, dims = n + 1))
        # errors
        yt .= selectdim(ε, n + 1, t)
        if t > lags(model)
            # autoregressive component
            for (p, Ap) in pairs(coef(model)), r in 1:rank(Ap)
                yt .+= loadings(Ap)[r] .* tucker(selectdim(y_burn, n + 1, t - p), U[p][r])
            end
        end
    end

    # simulate data
    y = similar(data(model))
    for (t, yt) in pairs(eachslice(y, dims = n + 1))
        # errors
        yt .= selectdim(ε, n + 1, burn + t)
        # autoregressive component
        for (p, Ap) in pairs(coef(model)), r in 1:rank(Ap)
            if t <= p
                ylag = selectdim(y_burn, n + 1, burn + t - p)
            else
                ylag = selectdim(y, n + 1, t - p)
            end
            yt .+= loadings(Ap)[r] .* tucker(ylag, U[p][r])
        end
    end

    # Kruskal coefficient
    A = [StaticKruskal((deepcopy(getproperty(Ap, p)) for p in propertynames(Ap))...)
         for Ap in coef(model)]

    return StaticTensorAutoregression(y, TensorNormal(deepcopy(cov(model))), A)
end
function simulate(model::DynamicTensorAutoregression; burn::Integer = 100,
                  rng::AbstractRNG = Xoshiro())
    d = dims(model)
    n = length(d)

    # Kruskal coefficient
    λ = simulate.(coef(model), nobs(model) + burn - lags(model), rng)
    A = [DynamicKruskal((deepcopy(getproperty(Ap, p)) for p in propertynames(Ap))...)
         for Ap in coef(model)]
    for (p, Ap) in pairs(A)
        loadings(Ap) .= λ[p][:, (burn + 1):(nobs(model) + burn - lags(model))]
    end

    # tensor error distribution
    ε = simulate(dist(model), nobs(model) + burn, rng)

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # burn-in
    y_burn = similar(data(model), d..., burn)
    for (t, yt) in pairs(eachslice(y_burn, dims = n + 1))
        # errors
        yt .= selectdim(ε, n + 1, t)
        if t > lags(model)
            # autoregressive component
            for (p, λp) in pairs(λ), (r, λpr) in pairs(eachrow(λp))
                yt .+= λpr[t - lags(model)] .*
                       tucker(selectdim(y_burn, n + 1, t - p), U[p][r])
            end
        end
    end

    # simulate data
    y = similar(data(model))
    for (t, yt) in pairs(eachslice(y, dims = n + 1))
        # errors
        yt .= selectdim(ε, n + 1, burn + t)
        # autoregressive component
        for (p, λp) in pairs(λ), (r, λpr) in pairs(eachrow(λp))
            if t <= p
                ylag = selectdim(y_burn, n + 1, burn + t - p)
            else
                ylag = selectdim(y, n + 1, t - p)
            end
            yt .+= λpr[burn - lags(model) + t] .* tucker(ylag, U[p][r])
        end
    end

    return DynamicTensorAutoregression(y, TensorNormal(deepcopy(cov(model))), A)
end

"""
    isstationary(model[, m = 100, samples = 250]) -> Bool

Test whether a tensor autoregressive model is stationary. In case the model is dynamic
stationarity is tested through the Lyaponov exponent.
"""
isstationary(model::StaticTensorAutoregression) = maximum(eigvals(companion(model)))
function isstationary(model::DynamicTensorAutoregression; m::Integer = 100,
                      samples::Integer = 250)
    return any(γ -> γ < 0, lyaponov(model, m, samples = samples))
end

"""
    fit!(model; init_method=(coef=:data, dist=:data), order=true, tolerance=1e-4,
         max_iter=1000, criteria=:objective, verbose=false) -> model

Fit the tensor autoregressive model described by `model` to the data with `tolerance` and
maximum number of iterations `max_iter`. `criteria` determines the criteria for convergence
and can be set to `:objective` for objective function value based or `:param` for parameter
value based. If `verbose` is true a summary of the model fitting is printed. `init_method`
indicates which method is used for initialization of the parameters.

When `order` is true, ordering ambiguity of the Kruskal rank components for each lag are
solved by ordering in decreasing fashion the rank components based on the persistence of the
respective loadings processes when `model` is dynamic or based on the static loadings if
`model` is static.

Estimation is done using the Expectation-Maximization algorithm for obtaining the maximum
likelihood estimates of the dynamic model and the alternating least squares (ALS) algorithm
for obtaining the least squares and maximum likelihood estimates of the static model, for
respectively white noise and tensor normal errors.
"""
function fit!(model::AbstractTensorAutoregression;
              init_method::NamedTuple = (coef = :data, dist = :data), order::Bool = true,
              tolerance::AbstractFloat = 1e-4, max_iter::Integer = 1000,
              criteria::Symbol = :objective, verbose::Bool = false)
    criteria ∈ (:objective, :param) ||
        error("convergence criteria must we either :objective or :param.")
    keys(init_method) ⊇ (:coef, :dist) ||
        error("init_method must be a NamedTuple with keys :coef and :dist.")

    # model summary
    if verbose
        println("Tensor autoregressive model")
        println("===========================")
        println("Dimensions: $(dims(model))")
        println("Number of observations: $(nobs(model))")
        println("Number of lags: $(lags(model))")
        println("Rank: $(rank(model))")
        println("Distribution: $(Base.typename(typeof(dist(model))).wrapper)")
        println("Coefficient tensor: ",
                coef(model) isa StaticKruskal ? "static" : "dynamic")
        println("===========================")
        println()
    end

    # initialization of model parameters
    init!(model, init_method)

    # alternating least squares
    iter = 0
    converged = false
    if criteria == :objective
        obj = -Inf
        violation = false
    elseif criteria == :param
        θ_prev = params(model)
        θ = copy(θ_prev)
    end
    while !converged && iter < max_iter
        # update model
        update!(model)

        if criteria == :objective
            # update objective function
            obj_prev = obj
            obj = objective(model)

            # non-decrease violation
            if obj - obj_prev < -1e-4
                violation = true
                if verbose
                    println("Objective function value decreased from $iter to $(iter + 1).")
                    println()
                end
                break
            end

            # delta
            δ = 2 * abs(obj - obj_prev) / (abs(obj) + abs(obj_prev))
        elseif criteria == :param
            # update parameters
            θ .= params(model)
            # delta
            δ = norm(θ - θ_prev)
            # store
            θ_prev .= θ
        end

        # convergence
        converged = δ < tolerance

        # update iteration counter
        iter += 1
    end

    # fix signs of Kruskal tensors
    fixsign!.(coef(model))

    # fix rank component ordering per lag
    if order
        # order rank components
        for Ap in coef(model)
            fixorder!(Ap)
        end
    end

    # optimization summary
    if verbose
        println("Optimization summary")
        println("====================")
        println("Convergence criteria: ", string(criteria))
        println("Convergence: ", converged ? "success" : "failed")
        criteria == :objective && println("Non-decrease violation: $violation")
        println("Iterations: $iter")
        println("Objective function value: $(objective(model))")
        println("aic: $(aic(model))")
        println("aicc: $(aicc(model))")
        println("bic: $(bic(model))")
        println("====================")
    end

    return model
end

"""
    forecast(model, periods[, samples = 1000, rng = Xoshiro()]) -> forecasts

Compute forecasts `periods` periods ahead using fitted tensor autoregressive model `model`,
using `samples` and random number generator `rng` for the Monte Carlo estimate of in the
dynamic case.
"""
function forecast(model::StaticTensorAutoregression, periods::Integer)
    d = dims(model)
    n = length(d)
    Ty = eltype(data(model))

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # forecast data using tensor autoregression
    forecasts = zeros(Ty, d..., periods)
    for h in 1:periods, (p, Ap) in pairs(coef(model))
        if h <= p
            yp = selectdim(data(model), n + 1, nobs(model) + h - p)
        else
            yp = selectdim(forecasts, n + 1, h - p)
        end
        for r in 1:rank(Ap)
            selectdim(forecasts, n + 1, h) .+= loadings(Ap)[r] * tucker(yp, U[p][r])
        end
    end

    return forecasts
end
function forecast(model::DynamicTensorAutoregression, periods::Integer;
                  samples::Integer = 1000, rng::AbstractRNG = Xoshiro())
    # sample conditional paths
    paths = sampler(model, samples, periods, nobs(model), rng)

    # Monte Carlo estimate
    return dropdims(mean(paths, dims = ndims(paths)), dims = ndims(paths))
end

"""
    irf(model, periods[, shock, index, t]; alpha = 0.05[, samples = 100, rng = Xoshiro()]) -> irfs

Compute generalized impulse response functions `periods` ahead and corresponding `alpha`%
upper and lower confidence bounds using fitted tensor autoregressive model `model`. Upper
and lower confidence bounds are computed using Monte Carlo simulation.

In case of a dynamic model `samples` and random number generator `rng` are used to produce a
Monte Carlo estimate for the generalized impulse response functions and `shock`, `index`,
and `t` are used to indicate the series to shock at a specific time point and with what
magnitude.

Note that Monte Carlo estimation of the generalized impulse response function for the
dynamic model has to be performed for each `index`, `shock` and time point individually. Due
to the high computational burden of this procedure generalized impulse response functions
for the dynamic model are only performed for a single time point at each function call.
"""
function irf(model::StaticTensorAutoregression, periods::Integer; alpha::Real = 0.05)
    # moving average representation
    Ψ = moving_average(model, periods)

    # girf
    Ψ_star = integrate(Ψ, cov(model))

    # confidence bounds
    (lower, upper) = confidence_bounds(model, periods, alpha)

    return StaticIRF(Ψ_star, lower, upper)
end
function irf(model::DynamicTensorAutoregression, periods::Integer, shock::Real, index::Dims,
             t::Int; alpha::Real = 0.05, samples::Integer = 100,
             rng::AbstractRNG = Xoshiro())
    # sample girfs
    Ψ_stars = similar(data(model), dims(model)..., periods, samples)
    for Ψ_star in eachslice(Ψ_stars, dims = ndims(Ψ_stars))
        # sample conditional paths
        conditional = sampler(model, samples, periods, t, shock, index, rng)

        # sample unconditional paths
        unconditional = sampler(model, samples, periods, t, rng)

        # Monte Carlo estimate of conditional expectations
        n = ndims(conditional)
        Ψ_star .= dropdims(mean(conditional, dims = n), dims = n) .-
                  dropdims(mean(unconditional, dims = n), dims = n)
    end

    # Monte Carlo estimate girf
    Ψ_star = dropdims(mean(Ψ_stars, dims = ndims(Ψ_stars)), dims = ndims(Ψ_stars))

    # quantiles
    lower_idx = round(Int, samples * alpha / 2)
    upper_idx = round(Int, samples * (1.0 - alpha / 2))

    # confidence bounds
    Ψ_sorted = sort(Ψ_stars, dims = ndims(Ψ_stars))
    lower = selectdim(Ψ_sorted, ndims(Ψ_sorted), lower_idx)
    upper = selectdim(Ψ_sorted, ndims(Ψ_sorted), upper_idx)

    return DynamicIRF(Ψ_star, lower, upper)
end
