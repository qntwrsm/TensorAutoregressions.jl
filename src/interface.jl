#=
interface.jl

    Provides a collection of interface tools for working with tensor autoregressive 
    models, such as estimation, forecasting, and impulse response analysis. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/03/02
=#

"""
    TensorAutoregression(
        y, 
        R; 
        dynamic=false, 
        dist=:white_noise,
        fixed=NamedTuple()
    ) -> model

Constructs a tensor autoregressive model for data `y` with autoregressive
coefficient tensor of rank `R`, potentially dynamic, and tensor error
distribution `dist` and fixed parameters indicated by `fixed`.
"""
function TensorAutoregression(
    y::AbstractArray, 
    R::Integer; 
    dynamic::Bool=false, 
    dist::Symbol=:white_noise,
    fixed::NamedTuple=NamedTuple()
)   
    dims = size(y)
    n = ndims(y) - 1

    # check model specification
    dynamic && dist == :white_noise && throw(ArgumentError("dynamic model with white noise error not supported."))

    # instantiate Kruskal autoregressive tensor
    if dynamic
        A = DynamicKruskal(
            similar(y, R, last(dims)-1), 
            similar(y, R), 
            Diagonal(similar(y, R)), 
            Symmetric(similar(y, R, R)),
            [similar(y, dims[i - n*((i-1)÷n)], R) for i = 1:2*n], 
            R
        )
    else
        A = StaticKruskal(
            similar(y, R), 
            [similar(y, dims[i - n*((i-1)÷n)], R) for i = 1:2*n], 
            R
        )
    end

    # instantiate tensor error distribution
    N = prod(dims[1:n])
    if dist == :white_noise
        ε = WhiteNoise(
            similar(y, dims[1:n]..., last(dims)-1), 
            Symmetric(similar(y, N, N))
        )
    elseif dist == :tensor_normal
        ε = TensorNormal(
            similar(y, dims[1:end-1]..., last(dims)-1), 
            [Symmetric(similar(y, dims[i], dims[i])) for i = 1:n]
        )
    else
        throw(ArgumentError("distribution $dist not supported."))
    end

    return TensorAutoregression(y, ε, A, fixed)
end

"""
    simulate(model; burn=100, rng=Xoshiro()) -> sim

Simulate data from the tensor autoregressive model described by `model` and
return a new instance with the simulated data, using random number generator
`rng` and apply a burn-in period of `burn`.
"""
function simulate(model::TensorAutoregression; burn::Integer=100, rng::AbstractRNG=Xoshiro())
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # Kruskal coefficient
    if coef(model) isa StaticKruskal
        A_sim = similar(coef(model))
        copyto!(A_sim, coef(model))
        A_burn = A_sim
    else
        (A_sim, A_burn) = simulate(coef(model), burn, rng)
    end

    # tensor error distribution
    (ε_sim, ε_burn) = simulate(dist(model), burn+1, rng)
    
    # outer product of Kruskal factors
    U = [[factors(A_sim)[i][:,r] * factors(A_sim)[i+n][:,r]' for i = 1:n] for r = 1:rank(A_sim)]

    # burn-in
    y_burn = similar(data(model), dims[1:n]..., burn+1)
    for (t, yt) ∈ pairs(eachslice(y_burn, dims=n+1))
        # errors
        yt .= selectdim(resid(ε_burn), n+1, t)
        if t > 1
            # autoregressive component
            for r = 1:rank(A_burn)
                λ = A_burn isa StaticKruskal ? loadings(A_burn)[r] : loadings(A_burn)[r,t-1]
                yt .+= λ .* tucker(selectdim(y_burn, n+1, t-1), U[r], 1:n)
            end
        end
    end

    # simulate data
    y_sim = similar(data(model))
    for (t, yt) ∈ pairs(eachslice(y_sim, dims=n+1))
        if t == 1
            yt .= selectdim(y_burn, n+1, burn+1)
        else
            # errors
            yt .= selectdim(resid(ε_sim), n+1, t-1)
            # autoregressive component
            for r = 1:rank(A_sim)
                λ = A_sim isa StaticKruskal ? loadings(A_sim)[r] : loadings(A_sim)[r,t-1]
                yt .+= λ .* tucker(selectdim(y_sim, n+1, t-1), U[r], 1:n)
            end
        end
    end

    return TensorAutoregression(y_sim, ε_sim, A_sim, fixed(model))
end

"""
    fit!(
        model;  
        init_method=(coef=:data, dist=:data), 
        ϵ=1e-4, 
        max_iter=1000, 
        verbose=false
    ) -> model

Fit the tensor autoregressive model described by `model` to the data with
tolerance `ϵ` and maximum number of iterations `max_iter`. If `verbose` is true
a summary of the model fitting is printed. `init_method` indicates which method
is used for initialization of the parameters.

Estimation is done using the Expectation-Maximization algorithm for
obtaining the maximum likelihood estimates of the dynamic model and the
alternating least squares (ALS) algorithm for obtaining the least squares and
maximum likelihood estimates of the static model, for respectively white noise
and tensor normal errors.
"""
function fit!(
    model::TensorAutoregression;
    init_method::NamedTuple=(coef=:data, dist=:data), 
    ϵ::AbstractFloat=1e-4, 
    max_iter::Integer=1000, 
    verbose::Bool=false
)
    rank(model) == 1 || error("general rank R model fitting not implemented.")
    keys(init_method) ⊇ (:coef, :dist) || error("init_method must be a NamedTuple with keys :coef and :dist.")

    # model summary
    if verbose
        println("Tensor autoregressive model")
        println("===========================")
        println("Dimensions: $(size(data(model))[1:end-1])")
        println("Number of observations: $(size(data(model))[end])")
        println("Rank: $(rank(model))")
        println("Distribution: $(Base.typename(typeof(dist(model))).wrapper)")
        println("Coefficient tensor: ", coef(model) isa StaticKruskal ? "static" : "dynamic")
        println("Fixed parameters:")
        println("   Kruskal tensor: ", haskey(fixed(model), :coef) ? keys(fixed(model).coef) : "none")
        println("   Distribution: ", haskey(fixed(model), :dist) ? keys(fixed(model).dist) : "none")
        println("===========================")
        println()
    end
    
    # initialization of model parameters
    init!(model, init_method)

    # instantiate model
    model_prev = copy(model)

    # alternating least squares
    iter = 0
    δ = Inf
    while δ > ϵ && iter < max_iter
        # update model
        update!(coef(model), dist(model), data(model), fixed(model))

        # compute maximum abs change in parameters
        δ = absdiff(model, model_prev)

        # store model
        copyto!(model_prev, model)

        # update iteration counter
        iter += 1
    end

    # optimization summary
    if verbose
        println("Optimization summary")
        println("====================")
        println("Convergence: ", δ < ϵ ? "success" : "failed")
        println("Maximum absolute change $δ")
        println("Iterations: $iter")
        println("Log-likelihood: $(loglike(model))")
        println("====================")
    end

    return model
end

"""
    forecast(model, periods) -> forecasts

Compute forecasts `periods` periods ahead using fitted tensor autoregressive
model `model`.
"""
function forecast(model::TensorAutoregression, periods::Integer)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # sample dynamic loadings particles
    if coef(model) isa DynamicKruskal
        particles = get_particles(data(model), coef(model), dist(model), periods)
    end

    # outer product of Kruskal factors
    U = [[factors(model)[i][:,r] * factors(model)[i+n][:,r]' for i = 1:n] for r = 1:rank(model)]

    # forecast data using tensor autoregression
    forecasts = similar(data(model), dims[1:n]..., periods)
    # last observation
    yT = selectdim(data(model), n+1, last(dims))
    # forecast
    for h = 1:periods, r = 1:rank(model)
        λ̂ = coef(model) isa StaticKruskal ? loadings(model)[r]^h : mean(prod(particles[r,:,1:h], dims=2))
        selectdim(forecasts, n+1, h) .= λ̂ * tucker(yT, U[r].^h, 1:n)
    end

    return forecasts
end

"""
    irf(model, periods; α=.05, orth=false) -> irfs

Compute impulse response functions `periods` periods ahead and corresponding
`α`% upper and lower confidence bounds using fitted tensor autoregressive model
`model`. Upper and lower confidence bounds are computed using Monte Carlo
simulation.
If `orth` is true, the orthogonalized impulse response functions are
computed.
"""
function irf(model::TensorAutoregression, periods::Integer; α::Real=.05, orth::Bool=false)
    # moving average representation
    if coef(model) isa StaticKruskal
        irf_type = StaticIRF
        Ψ = moving_average(coef(model), periods)
    else
        irf_type = DynamicIRF
        Ψ = moving_average(coef(model), periods, data(model), dist(model))
    end

    # orthogonalize
    orth ? Ψ = orthogonalize(Ψ, cov(model)) : nothing

    # confidence bounds
    (lower, upper) = confidence_bounds(model, periods, α, orth)

    return irf_type(Ψ, lower, upper, orth)
end