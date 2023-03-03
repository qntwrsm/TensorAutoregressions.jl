#=
interface.jl

    Provides a collection of interface tools for working with tensor autoregressive 
    models, such as estimation, forecasting, and impulse response analysis. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/03/02
=#

"""
    TensorAutoregression(y, R, dynamic=false, dist=:white_noise)

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
    # instantiate Kruskal autoregressive tensor
    if dynamic
        A = DynamicKruskal(
            similar(y, R), 
            Diagonal(similar(y, R)), 
            Symmetric(similar(y, R, R)),
            [similar(y, size(y, i), R) for i = 1:ndims(y)], 
            R
        )
    else
        A = StaticKruskal(
            similar(y, R), 
            [similar(y, size(y, i), R) for i = 1:ndims(y)], 
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