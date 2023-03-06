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
    # check model specification
    dynamic && dist == :white_noise || throw(ArgumentError("dynamic model with white noise error not supported."))

    # instantiate Kruskal autoregressive tensor
    if dynamic
        A = DynamicKruskal(
            similar(y, R, lastindex(y, ndims(y))), 
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
    N = prod(size(y)[1:end-1])
    if dist == :white_noise
        ε = WhiteNoise(similar(y), Symmetric(similar(y, N, N)))
    elseif dist == :tensor_normal
        ε = TensorNormal(similar(y), Symmetric(similar(y, N, N)))
    else
        throw(ArgumentError("distribution $dist not supported."))
    end

    return TensorAutoregression(y, ε, A)
end

"""
    fit!(model) -> model

Wrapper for fitting of tensor autoregressive model described by `model` to the
data. 

Estimation is done using the Expectation-Maximization algorithm for
obtaining the maximum likelihood estimates of the dynamic model and the
alternating least squares (ALS) algorithm for obtaining the least squares and
maximum likelihood estimates of the static model, for respectively white noise
and tensor normal errors.
"""
fit!(model::TensorAutoregression) = coef(model) isa DynamicKruskal ? em!(model) : als!(model)

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
    irf(model, periods, orth=false) -> irfs

Compute impulse response functions `periods` periods ahead using fitted tensor
autoregressive model `model`. If `orth` is true, the orthogonalized impulse
response functions are computed.
"""
function irf(model::TensorAutoregression, periods::Integer, orth::Bool=false)
    # TODO: implementation
end