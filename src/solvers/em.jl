#=
em.jl

    Provides expectation-maximization (EM) optimization routine for fitting a
    tensor autoregressive model with dynamic Kruskal coefficient tensor.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/08/02
=#

"""
    em!(model, ϵ, max_iter, verbose) -> model

Expectation-Maximization (EM) optimization routine for fitting a tensor
autoregressive model with a dynamic Kruskal coefficient tensor and tensor normal
errors given by `model` using maximum likelihood, with tolerance `ϵ` and maximum
number of iterations `max_iter`. Results are stored by overwritting `model`. If
verbose is `true`, a summary of the optimization is printed. 
"""
function em!(
    model::TensorAutoregression, 
    ϵ::AbstractFloat, 
    max_iter::Integer, 
    verbose::Bool
)
    # initialization of model parameters
    init!(model)

    # instantiate model
    model_prev = copy(model)

    # Expectation-Maximization
    iter = 0
    δ = Inf
    while δ > ϵ && iter < max_iter
        # update model
        update!(model)

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