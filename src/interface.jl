#=
interface.jl

    Provides a collection of interface tools for working with tensor autoregressive models,
    such as estimation, forecasting, and impulse response analysis.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/03/02
=#

"""
    TensorAutoregression(y, R; dynamic=false, dist=:white_noise) -> model

Constructs a tensor autoregressive model for data `y` with autoregressive coefficient tensor
of rank `R`, potentially dynamic, and tensor error distribution `dist`.
"""
function TensorAutoregression(y::AbstractArray, R::Integer; dynamic::Bool = false,
                              dist::Symbol = :white_noise)
    dims = size(y)
    n = ndims(y) - 1

    # check model specification
    dynamic && dist == :white_noise &&
        throw(ArgumentError("dynamic model with white noise error not supported."))

    # instantiate Kruskal autoregressive tensor
    if dynamic
        A = DynamicKruskal(similar(y, R, last(dims) - 1), similar(y, R),
                           Diagonal(similar(y, R)), Diagonal(similar(y, R)),
                           [similar(y, dims[i - n * ((i - 1) ÷ n)], R) for i in 1:(2n)], R)
        model = DynamicTensorAutoregression
    else
        A = StaticKruskal(similar(y, R),
                          [similar(y, dims[i - n * ((i - 1) ÷ n)], R) for i in 1:(2n)], R)
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
coefficient tensor of rank `R`, potentially dynamic, and tensor error distribution `dist`.
"""
function TensorAutoregression(dims::Dims, R::Integer; dynamic::Bool = false,
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

    # Kruskal coefficient
    A_sim = StaticKruskal((copy(getproperty(coef(model), p)) for p in propertynames(coef(model)))...)

    # tensor error distribution
    (ε_sim, ε_burn) = simulate(dist(model), burn + 1, rng)

    # outer product of Kruskal factors
    U = outer(coef(model))

    # burn-in
    y_burn = similar(data(model), dims[1:n]..., burn + 1)
    for (t, yt) in pairs(eachslice(y_burn, dims = n + 1))
        # errors
        yt .= selectdim(resid(ε_burn), n + 1, t)
        if t > 1
            # autoregressive component
            for r in 1:rank(A_sim)
                yt .+= loadings(A_sim)[r] .* tucker(selectdim(y_burn, n + 1, t - 1), U[r])
            end
        end
    end

    # simulate data
    y_sim = similar(data(model))
    for (t, yt) in pairs(eachslice(y_sim, dims = n + 1))
        if t == 1
            yt .= selectdim(y_burn, n + 1, burn + 1)
        else
            # errors
            yt .= selectdim(resid(ε_sim), n + 1, t - 1)
            # autoregressive component
            for r in 1:rank(A_sim)
                yt .+= loadings(A_sim)[r] .* tucker(selectdim(y_sim, n + 1, t - 1), U[r])
            end
        end
    end

    return StaticTensorAutoregression(y_sim, ε_sim, A_sim)
end
function simulate(model::DynamicTensorAutoregression; burn::Integer = 100,
                  rng::AbstractRNG = Xoshiro())
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # Kruskal coefficient
    (A_sim, A_burn) = simulate(coef(model), burn, rng)

    # tensor error distribution
    (ε_sim, ε_burn) = simulate(dist(model), burn + 1, rng)

    # outer product of Kruskal factors
    U = outer(coef(model))

    # burn-in
    y_burn = similar(data(model), dims[1:n]..., burn + 1)
    for (t, yt) in pairs(eachslice(y_burn, dims = n + 1))
        # errors
        yt .= selectdim(resid(ε_burn), n + 1, t)
        if t > 1
            # autoregressive component
            for r in 1:rank(A_burn)
                yt .+= loadings(A_burn)[r, t - 1] .*
                       tucker(selectdim(y_burn, n + 1, t - 1), U[r])
            end
        end
    end

    # simulate data
    y_sim = similar(data(model))
    for (t, yt) in pairs(eachslice(y_sim, dims = n + 1))
        if t == 1
            yt .= selectdim(y_burn, n + 1, burn + 1)
        else
            # errors
            yt .= selectdim(resid(ε_sim), n + 1, t - 1)
            # autoregressive component
            for r in 1:rank(A_sim)
                yt .+= loadings(A_sim)[r, t - 1] .*
                       tucker(selectdim(y_sim, n + 1, t - 1), U[r])
            end
        end
    end

    return DynamicTensorAutoregression(y_sim, ε_sim, A_sim)
end

"""
    fit!(model; init_method=(coef=:data, dist=:data), ϵ=1e-4, max_iter=1000, verbose=false)
        -> model

Fit the tensor autoregressive model described by `model` to the data with tolerance `ϵ` and
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
    forecast(model, periods) -> forecasts

Compute forecasts `periods` periods ahead using fitted tensor autoregressive
model `model`.
"""
function forecast(model::StaticTensorAutoregression, periods::Integer)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # outer product of Kruskal factors
    U = outer(coef(model))

    # forecast data using tensor autoregression
    forecasts = similar(data(model), dims[1:n]..., periods)
    # last observation
    yT = selectdim(data(model), n + 1, last(dims))
    # forecast
    for h in 1:periods, r in 1:rank(model)
        selectdim(forecasts, n + 1, h) .= loadings(model)[r]^h * tucker(yT, U[r] .^ h)
    end

    return forecasts
end
function forecast(model::DynamicTensorAutoregression, periods::Integer)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # sample dynamic loadings particles
    particles = particle_sampler(model, periods)

    # outer product of Kruskal factors
    U = outer(coef(model))

    # forecast data using tensor autoregression
    forecasts = similar(data(model), dims[1:n]..., periods)
    # last observation
    yT = selectdim(data(model), n + 1, last(dims))
    # forecast
    for h in 1:periods, r in 1:rank(model)
        selectdim(forecasts, n + 1, h) .= mean(prod(particles[r, :, 1:h], dims = 2)) *
                                          tucker(yT, U[r] .^ h)
    end

    return forecasts
end

"""
    irf(model, periods; alpha=.05, orth=false) -> irfs

Compute impulse response functions `periods` periods ahead and corresponding `alpha`% upper
and lower confidence bounds using fitted tensor autoregressive model `model`. Upper and
lower confidence bounds are computed using Monte Carlo simulation. If `orth` is true, the
orthogonalized impulse response functions are computed.
"""
function irf(model::AbstractTensorAutoregression, periods::Integer; alpha::Real = 0.05,
             orth::Bool = false)
    if model isa StaticTensorAutoregression
        irf_type = StaticIRF
    else
        irf_type = DynamicIRF
    end

    # moving average representation
    Ψ = moving_average(model, periods)

    # orthogonalize
    orth ? Ψ = orthogonalize(Ψ, cov(model)) : nothing

    # confidence bounds
    (lower, upper) = confidence_bounds(model, periods, alpha, orth)

    return irf_type(Ψ, lower, upper, orth)
end
