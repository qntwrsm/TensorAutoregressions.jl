#=
TensorAutoregressions.jl

    Provides a collection of tools for working with tensor autoregressive 
    models, such as estimation, forecasting, and impulse response analysis. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/02/27
=#

module TensorAutoregressions

using LinearAlgebra

export TensorAutoregression

include("tensor_algebra.jl")
include("types.jl")
include("interface.jl")

end
