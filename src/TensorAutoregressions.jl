#=
TensorAutoregressions.jl

    Provides a collection of tools for working with tensor autoregressive 
    models, such as estimation, forecasting, and impulse response analysis. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/02/27
=#

module TensorAutoregressions

using 
    LinearAlgebra,
    Statistics, 
    Random, 
    Distributions, 
    TensorToolbox, 
    Optim, 
    CairoMakie,
    Dates

import LinearAlgebra: rank
import Statistics: cov

# Makie backend and theme
CairoMakie.activate!()

export
    # constructor
    TensorAutoregression,

    # interface methods
    ## getters
    data, coef, dist,   # general
    resid, cov, # residuals
    factors, loadings, rank, intercept, dynamics, # Kruskal coefficient tensor

    # simulate
    simulate,

    ## fit
    fit!,

    ## forecast
    forecast,

    ## impulse response functions
    irf,
    lower, upper, orth,

    # plotting
    data_plot,
    kruskal_plot,
    cov_plot,
    irf_plot

include("tensor_algebra.jl")
include("types.jl")
include("interface.jl")
include("utilities.jl")
include("fit/utilities.jl")
include("fit/solver.jl")
include("plotting.jl")

end
