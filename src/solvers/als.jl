#=
als.jl

    Provides alternating least squares (ALS) optimization routine for fitting a 
    tensor autoregressive model with static Kruskal coefficient tensor. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/08/02
=#

"""
    update_coef!(model)

Update Kruskal coefficient tensor.
"""
function update_coef!(model::TensorAutoregression)
    # TODO: implementation
    return nothing
end

"""
    update_cov!(model)

Update tensor error distribution covariance.
"""
function update_cov!(model::TensorAutoregression)
    # TODO: implementation
    return nothing
end

"""
    als!(model, ϵ, max_iter, verbose) -> model

Alternating least squares (ALS) optimization routine for fitting a tensor
autoregressive model with a static Kruskal coefficient tensor given by `model`,
with tolerance `ϵ` and maximum number of iterations `max_iter`. Results are
stored by overwritting `model`. If verbose is `true`, a summary of the
optimization is printed. 
"""
function als!(
    model::TensorAutoregression, 
    ϵ::AbstractFloat, 
    max_iter::Integer, 
    verbose::Bool
)
    # initialization of model parameters
    init!(model)

    # instantiate model
    model_prev = similar(model)

    # alternating least squares
    iter = 0
    δ = Inf
    while δ > ϵ && iter < max_iter
        # store model
        copyto!(model_prev, model)

        # update Kruskal coefficient tensor
        update_coef!(model)

        # update tensor error distribution covariance
        update_cov!(model)

        # compute maximum abs change in parameters
        δ = absdiff(model, model_prev)

        # update iteration counter
        iter += 1
    end

    # optimization summary
    if verbose
        println("optimization summary not implemented.")
    end

    return model
end