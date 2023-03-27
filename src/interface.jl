#=
interface.jl

    Provides a collection of interface tools for working with tensor autoregressive 
    models, such as estimation, forecasting, and impulse response analysis. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/03/02
=#

"""
    TensorAutoregression(y, R, dynamic=false, dist=:white_noise) -> model

Constructs a tensor autoregressive model for data `y` with autoregressive
coefficient tensor of rank `R`, potentially dynamic, and tensor error
distribution `dist`.
"""
function TensorAutoregression(
    y::AbstractArray, 
    R::Integer, 
    dynamic::Bool=false, 
    dist::Symbol=:white_noise
)   
    dims = size(y)

    # check model specification
    dynamic && dist == :white_noise && throw(ArgumentError("dynamic model with white noise error not supported."))

    # instantiate Kruskal autoregressive tensor
    if dynamic
        A = DynamicKruskal(
            similar(y, R, last(dims)-1), 
            Diagonal(similar(y, R)), 
            Symmetric(similar(y, R, R)),
            [similar(y, size(y, i), R) for i = 1:ndims(y)-1], 
            R
        )
    else
        A = StaticKruskal(
            similar(y, R), 
            [similar(y, size(y, i), R) for i = 1:ndims(y)-1], 
            R
        )
    end

    # instantiate tensor error distribution
    N = prod(dims[1:end-1])
    if dist == :white_noise
        ε = WhiteNoise(
            similar(y, dims[1:end-1]..., last(dims)-1), 
            Symmetric(similar(y, N, N))
        )
    elseif dist == :tensor_normal
        ε = TensorNormal(
            similar(y, dims[1:end-1]..., last(dims)-1), 
            Symmetric(similar(y, N, N))
        )
    else
        throw(ArgumentError("distribution $dist not supported."))
    end

    return TensorAutoregression(y, ε, A)
end

"""
    simulate(model, rng=Xoshiro()) -> model_sim

Simulate data from the tensor autoregressive model described by `model` and
return a new instance with the simulated data, using random number generator
`rng`.
"""
function simulate(model::TensorAutoregression, rng::AbstractRNG=Xoshiro())
    # Kruskal coefficient
    if isa(coef(model), StaticKruskal)
        A_sim = similar(coef(model))
        copyto!(A_sim, coef(model))
    else
        A_sim = simulate(coef(model), rng)
    end

    # tensor error distribution
    ε_sim = simulate(dist(model), rng)

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(ε_sim)))
    
    # outer product of Kruskal factors
    U = [factors(model)[i] * factors(model)[i+n]' for i = 1:n]

    # simulate data
    y_sim = similar(data(model))
    for (t, yt) ∈ pairs(eachslice(y_sim, dims=n+1))
        if t == 1
            # initial condition
            yt .= tucker(randn(rng, dims[1:end-1]...), C, 1:n)
        else
            # errors
            yt .= selectdim(resid(model), n+1, t-1)
            # autoregressive component
            for r = 1:rank(model)
                if isa(coef(model), StaticKruskal)
                    yt .+= loadings(model)[r] .* tucker(selectdim(y_sim, n+1, t-1), U, 1:n)
                elseif isa(coef(model), DynamicKruskal)
                    yt .+= loadings(model)[r,t] .* tucker(selectdim(y_sim, n+1, t-1), U, 1:n)
                end
            end
        end
    end

    return TensorAutoregression(y_sim, A_sim, ε_sim)
end

"""
    fit!(model, ϵ=1e-4, max_iter=1e3, verbose=false) -> model

Fit the tensor autoregressive model described by `model` to the data with
tolerance `ϵ` and maximum number of iterations `max_iter`. If `verbose` is true
a summary of the model fitting is printed.

Estimation is done using the Expectation-Maximization algorithm for
obtaining the maximum likelihood estimates of the dynamic model and the
alternating least squares (ALS) algorithm for obtaining the least squares and
maximum likelihood estimates of the static model, for respectively white noise
and tensor normal errors.
"""
function fit!(
    model::TensorAutoregression, 
    ϵ::AbstractFloat=1e-4, 
    max_iter::Integer=1e3, 
    verbose::Bool=false
)
    rank(model) != 1 || error("general rank R model fitting not implemented.")
    
    # initialization of model parameters
    init!(model)

    # instantiate model
    model_prev = copy(model)

    # alternating least squares
    iter = 0
    δ = Inf
    while δ > ϵ && iter < max_iter
        # update model
        update!(coef(model), dist(model), data(model))

        # compute maximum abs change in parameters
        δ = absdiff(model, model_prev)

        # store model
        copyto!(model_prev, model)

        # update iteration counter
        iter += 1
    end

    # optimization summary
    if verbose
        println("optimization summary not implemented.")
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

    # forecast dynamic coefficients
    if coef(model) isa DynamicKruskal
        # TODO: implementation
        error("dynamic coefficient forecasts not implemented.")
    end

    # forecast data using tensor autoregression
    forecasts = similar(data(model), dims[1:end-1]..., periods)
    for h = 1:periods
        if h == 1
            # last observation
            X = selectdim(data(model), ndims(data(model)), last(dims))
        else
            # previous forecast
            X = selectdim(forecasts, ndims(forecasts), h - 1)
        end
        # forecast
        selectdim(forecasts, ndims(forecasts), h) .= coef(model) * X
    end

    return forecasts
end

"""
    irf(model, periods, orth=false) -> Ψ

Compute impulse response functions `periods` periods ahead using fitted tensor
autoregressive model `model`. If `orth` is true, the orthogonalized impulse
response functions are computed.
"""
function irf(model::TensorAutoregression, periods::Integer, orth::Bool=false)
    # moving average representation
    Ψ = moving_average(coef(model), periods)

    # orthogonalize
    orth ? Ψ = orthogonalize(Ψ, cov(model)) : nothing

    return Ψ
end