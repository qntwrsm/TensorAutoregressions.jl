#=
TensorAutoregressions.jl

    Provides a collection of tools for working with tensor autoregressive 
    models, such as estimation, forecasting, and impulse response analysis. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/02/27
=#

module TensorAutoregressions

using LinearAlgebra, Random, Statistics, Distributions, TensorToolbox

export
    # main type + constructor
    TensorAutoregression,

    # interface methods
    ## getters
    data, coef, dist,   # general
    resid, cov, # residuals
    factors, loadings, rank, # Kruskal coefficient tensor

    # simulate
    simulate,

    ## fit
    fit!,

    ## forecast
    forecast,

    ## impulse response functions
    irf

include("tensor_algebra.jl")
include("types.jl")
include("interface.jl")
include("utilities.jl")
include("fit/utilities.jl")
include("fit/solver.jl")

end
