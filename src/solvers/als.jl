#=
als.jl

    Provides alternating least squares (ALS) optimization routine for fitting a 
    tensor autoregressive model with static Kruskal coefficient tensor. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/08/02
=#

"""
    update!(model)

Update Kruskal coefficient tensor and tensor error distribution covariance for
the tensor autoregressive model.
"""
function update!(model::TensorAutoregression)
    dims = size(data(model))
    n = ndims(data(model)) - 1

    # scaling
    if dist(model) isa WhiteNoise
        # identity matrices
        Cinv = [I for _ = 1:n]
    elseif dist(model) isa TensorNormal
        # Cholesky decompositions of Σᵢ
        C = cholesky.(Hermitian.(cov(model)))
        # inverse of Cholesky decompositions
        Cinv = [inv(C[i].L) for i = 1:n]
    end
    # inverse scaling
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = [factors(model)[i] * factors(model)[i+n]' for i = 1:n]
    if dist(model) isa TensorNormal
        # multiply with inverse Cholesky
        for i = 1:n
            lmul!(Cinv[i].L, U[i])
        end
    end

    # lag and lead variables
    y_lead = selectdim(data(model), n+1, 2:last(dims))
    y_lag = selectdim(data(model), n+1, 1:last(dims)-1)

    for k = 1:n
        m = setdiff(1:n, k)
        # matricize dependent variable along k-th mode
        Z = tucker(y_lead, Cinv[m], m)
        Zk = matricize(Z, k)
        # matricize regressor along k-th mode
        X = tucker(y_lag, U[m], m)
        Xk = matricize(X, k)

        # Gram matrix
        G = Xk * Xk'
        # moment matrix
        M = Zk * Xk'

        # update factor k
        factors(model)[k] .= M * factors(model)[k+n]
        factors(model)[k] .*= inv(loadings(model)[1] * dot(factors(model)[k+n], G, factors(model)[k+n]))
        # normalize
        factors(model)[k] .*= inv(norm(factors(model)[k]))
        # update factor k+n
        factors(model)[k+n] .= inv(G) * M' * Ω[k] * factors(model)[k]
        factors(model)[k+n] .*= inv(loadings(model)[1] * dot(factors(model)[k], Ω[k], factors(model)[k]))
        # normalize
        factors(model)[k+n] .*= inv(norm(factors(model)[k+n]))

        # update outer product of Kruskal factors
        U[k] .= factors(model)[k] * factors(model)[k+n]'

        if dist(model) isa TensorNormal
            # update residuals
            resid(model) .= Z .- loadings(model)[1] * tucker(X, U[k], k)

            # update covariance
            Ek = matricize(resid(model), k)
            mul!(cov(model).data[k], Ek, Ek', inv((last(dims) - 1) * prod(dims[m])), .0)
        
            # update multiply with inverse Cholesky
            lmul!(Cinv[k].L, U[k])
        end
    end

    # dependent variable and regressor tensors
    Z = tucker(y_lead, Cinv, 1:n)
    X = tucker(y_lag, U, 1:n)

    # update loading
    loadings(model)[1] = dot(Z, X) * inv(norm(X)^2)

    # update residuals
    resid(model) .= Z .- loadings(model)[1] .* X

    if dist(model) isa WhiteNoise
        # update covariance
        E = matricize(resid(model), 1:n)
        mul!(cov(model).data, E, E')
    end

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
    model_prev = copy(model)

    # alternating least squares
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