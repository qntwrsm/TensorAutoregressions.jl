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
function TensorAutoregression(y::AbstractArray, R::AbstractArray; dynamic::Bool = false,
                              dist::Symbol = :white_noise)
    dims = size(y)
    n = ndims(y) - 1
    p = length(R)

    # check model specification
    dynamic && dist == :white_noise &&
        throw(ArgumentError("dynamic model with white noise error not supported."))

    # instantiate Kruskal autoregressive tensors
    if dynamic
        A = [DynamicKruskal(similar(y, Rp, last(dims) - p), similar(y, Rp),
                            Diagonal(similar(y, Rp)), Diagonal(similar(y, Rp)),
                            [similar(y, dims[i - n * ((i - 1) ÷ n)], Rp) for i in 1:(2n)],
                            Rp) for Rp in R]
        model = DynamicTensorAutoregression
    else
        A = [StaticKruskal(similar(y, Rp),
                           [similar(y, dims[i - n * ((i - 1) ÷ n)], Rp) for i in 1:(2n)],
                           Rp) for Rp in R]
        model = StaticTensorAutoregression
    end

    # instantiate tensor error distribution
    N = prod(dims[1:n])
    if dist == :white_noise
        ε = WhiteNoise(Symmetric(similar(y, N, N)))
    elseif dist == :tensor_normal
        ε = TensorNormal([Symmetric(similar(y, dims[i], dims[i])) for i in 1:n])
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
function TensorAutoregression(dims::Dims, R::AbstractArray; dynamic::Bool = false,
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
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # tensor error distribution
    ε = simulate(dist(model), last(dims) + burn, rng)

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # burn-in
    y_burn = similar(data(model), dims[1:n]..., burn + lags(model))
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
        if t <= lags(model)
            yt .= selectdim(y_burn, n + 1, burn + t)
        else
            # errors
            yt .= selectdim(ε, n + 1, burn + t)
            # autoregressive component
            for (p, Ap) in pairs(coef(model)), r in 1:rank(Ap)
                yt .+= loadings(Ap)[r] .* tucker(selectdim(y, n + 1, t - p), U[p][r])
            end
        end
    end

    # Kruskal coefficient
    A = [StaticKruskal((copy(getproperty(Ap, p)) for p in propertynames(Ap))...)
         for Ap in coef(model)]

    return StaticTensorAutoregression(y, TensorNormal(copy(cov(model))), A)
end
function simulate(model::DynamicTensorAutoregression; burn::Integer = 100,
                  rng::AbstractRNG = Xoshiro())
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # Kruskal coefficient
    λ = simulate.(coef(model), last(dims) + burn, rng)
    A = [DynamicKruskal((copy(getproperty(Ap, p)) for p in propertynames(Ap))...)
         for Ap in coef(model)]
    for (p, Ap) in pairs(A)
        loadings(Ap) .= λ[p][:, (burn + lags(model) + 1):(last(dims) + burn)]
    end

    # tensor error distribution
    ε = simulate(dist(model), last(dims) + burn, rng)

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # burn-in
    y_burn = similar(data(model), dims[1:n]..., burn + lags(model))
    for (t, yt) in pairs(eachslice(y_burn, dims = n + 1))
        # errors
        yt .= selectdim(ε, n + 1, t)
        if t > lags(model)
            # autoregressive component
            for (p, λp) in pairs(λ), (r, λpr) in pairs(eachrow(λp))
                yt .+= λpr[t] .* tucker(selectdim(y_burn, n + 1, t - p), U[p][r])
            end
        end
    end

    # simulate data
    y = similar(data(model))
    for (t, yt) in pairs(eachslice(y, dims = n + 1))
        if t <= lags(model)
            yt .= selectdim(y_burn, n + 1, burn + t)
        else
            # errors
            yt .= selectdim(ε, n + 1, burn + t)
            # autoregressive component
            for (p, λp) in pairs(λ), (r, λpr) in pairs(eachrow(λp))
                yt .+= λpr[burn + t] .* tucker(selectdim(y, n + 1, t - p), U[p][r])
            end
        end
    end

    return DynamicTensorAutoregression(y, TensorNormal(copy(cov(model))), A)
end

"""
    fit!(model; init_method=(coef=:data, dist=:data), tolerance=1e-4, max_iter=1000,
         verbose=false) -> model

Fit the tensor autoregressive model described by `model` to the data with `tolerance` and
maximum number of iterations `max_iter`. If `verbose` is true a summary of the model fitting
is printed. `init_method` indicates which method is used for initialization of the
parameters.

Estimation is done using the Expectation-Maximization algorithm for obtaining the maximum
likelihood estimates of the dynamic model and the alternating least squares (ALS) algorithm
for obtaining the least squares and maximum likelihood estimates of the static model, for
respectively white noise and tensor normal errors.
"""
function fit!(model::AbstractTensorAutoregression;
              init_method::NamedTuple = (coef = :data, dist = :data),
              tolerance::AbstractFloat = 1e-4, max_iter::Integer = 1000,
              verbose::Bool = false)
    keys(init_method) ⊇ (:coef, :dist) ||
        error("init_method must be a NamedTuple with keys :coef and :dist.")

    # model summary
    if verbose
        println("Tensor autoregressive model")
        println("===========================")
        println("Dimensions: $(size(data(model))[1:end-1])")
        println("Number of observations: $(size(data(model))[end])")
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
    obj = -Inf
    converged = false
    violation = false
    while !converged && iter < max_iter
        # update model
        update!(model)

        # update objective function
        obj_prev = obj
        obj = objective(model)

        # non-decrease violation
        if obj - obj_prev < 0
            violation = true
            if verbose
                println("Objective function value decreased from $iter to $(iter + 1).")
                println()
            end
            break
        end

        # convergence
        δ = 2 * abs(obj - obj_prev) / (abs(obj) + abs(obj_prev))
        converged = δ < tolerance

        # update iteration counter
        iter += 1
    end

    # optimization summary
    if verbose
        println("Optimization summary")
        println("====================")
        println("Convergence: ", converged ? "success" : "failed")
        println("Non-decrease violation: $violation")
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
    forecast(model, periods[, samples=1000]) -> forecasts

Compute forecasts `periods` periods ahead using fitted tensor autoregressive model `model`,
using `samples` for the Monte Carlo estimate of in the dynamic case.
"""
function forecast(model::StaticTensorAutoregression, periods::Integer)
    dims = size(data(model))
    n = ndims(data(model)) - 1
    Ty = eltype(data(model))

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # forecast data using tensor autoregression
    forecasts = zeros(Ty, dims[1:n]..., periods)
    for h in 1:periods, (p, Ap) in pairs(coef(model))
        if h <= p
            yp = selectdim(data(model), n + 1, last(dims) + h - p)
        else
            yp = selectdim(forecasts, n + 1, h - p)
        end
        for r in 1:rank(Ap)
            selectdim(forecasts, n + 1, h) .+= loadings(Ap)[r] * tucker(yp, U[p][r])
        end
    end

    return forecasts
end
function forecast(model::DynamicTensorAutoregression, periods::Integer,
                  samples::Integer = 1000)
    dims = size(data(model))

    # sample conditional paths
    paths = sampler(model, samples, periods, last(dims))

    # Monte Carlo estimate
    return dropdims(mean(paths, dims = ndims(paths)), dims = ndims(paths))
end

"""
    irf(model, periods; alpha=0.05[, samples=100]) -> irfs

Compute generalized impulse response functions `periods` ahead and corresponding `alpha`%
upper and lower confidence bounds using fitted tensor autoregressive model `model`. Upper
and lower confidence bounds are computed using Monte Carlo simulation. In case of a dynamic
model `samples` are used to produce a Monte Carlo estimate for the generalized impulse
response functions.
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
function irf(model::DynamicTensorAutoregression, periods::Integer; alpha::Real = 0.05,
             samples::Integer = 100)
    Ψ_stars = map(1:samples) do _
        # sample conditional paths
        cond = sampler(model, samples, periods, σ, index)

        # sample unconditional paths
        uncond = sampler(model, samples, periods)

        # Monte Carlo estimate of conditional expectations
        dropdims(mean(cond, dims = ndims(cond)), dims = ndims(cond)) .-
        dropdims(mean(uncond, dims = ndims(uncond)), dims = ndims(uncond))
    end
    Ψ_stars = cat(Ψ_stars..., dims = ndims(Ψ_stars[1]) + 1)

    # Monte Carlo estimate
    Ψ_star = mean(Ψ_stars, dims = ndims(Ψ_stars))

    # quantiles
    lower_idx = round(Int, samples * α / 2)
    upper_idx = round(Int, samples * (1.0 - α / 2))

    # confidence bounds
    Ψ_sorted = sort(Ψ_stars, dims = ndims(Ψ_stars))
    lower = selectdim(Ψ_sorted, ndims(Ψ_sorted), lower_idx)
    upper = selectdim(Ψ_sorted, ndims(Ψ_sorted), upper_idx)

    return DynamicIRF(Ψ_star, lower, upper)
end
