#=
solver.jl

    Provides alternating least squares (ALS) and expectation-maximization (EM) optimization
    routines for fitting a tensor autoregressive model with Kruskal coefficient tensor and
    tensor error distribution.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/08/02
=#

"""
    update!(model)

Update parameters of tensor autoregression `model` using an alternating least squares (ALS)
solve. Wrapper to invoke multiple dispatch over static and dynamic tensor autoregressions.
"""
update!(model::StaticTensorAutoregression) = als!(coef(model), dist(model), data(model))
function update!(model::DynamicTensorAutoregression)
    # E-step
    # smoother
    (α, V, Γ) = smoother(model)
    loadings(model) .= hcat(α...)

    # M-step
    update_transition!(coef(model), V, Γ)
    als!(coef(model), dist(model), data(model), V)

    return nothing
end

"""
    als!(A, ε, y[, V])

Update Kruskal coefficient tensor `A` and tensor error distribution `ε` for the tensor
autoregressive model based on data `y` and smoothed loading variance `V` when `A` is dynamic
using an alternating least squares (ALS) solve.
"""
function als!(A::StaticKruskal, ε::WhiteNoise, y::AbstractArray)
    dims = size(y)
    n = ndims(y) - 1

    # outer product of Kruskal factors
    U = outer(A)

    # lag and lead variables
    y_lead = selectdim(y, n + 1, 2:last(dims))
    y_lag = selectdim(y, n + 1, 1:(last(dims) - 1))

    # initialize residuals
    resid = copy(y_lead)

    for r in 1:rank(A)
        s = setdiff(1:rank(A), r)
        # dependent variable tensor
        Zr = copy(y_lead)
        for i in s
            Zr .-= loadings(A)[i] .* tucker(y_lag, U[r][i])
        end
        for k in 1:n
            m = setdiff(1:n, k)
            # matricize dependent variable along k-th mode
            Zkr = matricize(Zr, k)
            # matricize regressor along k-th mode
            Xr = tucker(y_lag, U[r][m], m)
            Xkr = matricize(Xr, k)

            # Gram matrix
            G = Xkr * Xkr'
            # moment matrix
            M = Zkr * Xkr'

            # update factor k
            update_factor!(factors(A)[k][:, r], factors(A)[k + n][:, r], G \ M',
                           inv(loadings(A)[r]))
            # update factor k+n
            update_factor!(factors(A)[k + n][:, r], factors(A)[k][:, r], M,
                           inv(loadings(A)[r] *
                               dot(factors(A)[k][:, r], G, factors(A)[k][:, r])))

            # update outer product of Kruskal factors
            U[r][k] = factors(A)[k + n][:, r] * factors(A)[k][:, r]'
        end

        # regressor tensor
        Xr = tucker(y_lag, U[r])

        # update loading
        loadings(A)[r] = dot(Zr, Xr) / norm(Xr)^2

        # update residuals
        resid .-= loadings(A)[r] .* Xr
    end

    # update covariance
    E = matricize(resid, 1:n)
    mul!(cov(ε).data, E, E')

    return nothing
end
function als!(A::StaticKruskal, ε::TensorNormal, y::AbstractArray)
    dims = size(y)
    n = ndims(y) - 1

    # Cholesky decompositions of Σᵢ
    C = cholesky.(cov(ε))
    # inverse of Cholesky decompositions
    Cinv = inv.(getproperty.(C, :L))
    # precision matrices Ωᵢ
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = outer(A)

    # scaling
    S = [[Cinv[i] * U[r][i] for i in 1:n] for r in 1:rank(A)]

    # lag and lead variables
    y_lead = selectdim(y, n + 1, 2:last(dims))
    y_lag = selectdim(y, n + 1, 1:(last(dims) - 1))

    for r in 1:rank(A)
        s = setdiff(1:rank(A), r)
        # dependent variable tensor
        Zr = copy(y_lead)
        for i in s
            Xi = tucker(y_lag, U[i])
            Zr .-= loadings(A)[i] .* Xi
        end
        for k in 1:n
            m = setdiff(1:n, k)
            # matricize dependent variable along k-th mode
            Zr_scaled = tucker(Zr, Cinv[m], m)
            Zkr = matricize(Zr_scaled, k)
            # matricize regressor along k-th mode
            Xr = tucker(y_lag, S[r][m], m)
            Xkr = matricize(Xr, k)

            # Gram matrix
            G = Xkr * Xkr'
            # moment matrix
            M = Zkr * Xkr'

            # update factor k
            update_factor!(factors(A)[k][:, r], factors(A)[k + n][:, r], G \ M' * Ω[k],
                           inv(loadings(A)[r] *
                               dot(factors(A)[k + n][:, r], Ω[k], factors(A)[k + n][:, r])))
            # update factor k+n
            update_factor!(factors(A)[k + n][:, r], factors(A)[k][:, r], M,
                           inv(loadings(A)[r] *
                               dot(factors(A)[k][:, r], G, factors(A)[k][:, r])))

            # update outer product of Kruskal factors
            U[r][k] = factors(A)[k + n][:, r] * factors(A)[k][:, r]'

            # update covariance
            Ek = Zkr - loadings(A)[r] .* U[r][k] * Xkr
            mul!(cov(ε)[k].data, Ek, Ek', inv((last(dims) - 1) * prod(dims[m])), 0.0)
            # normalize
            k != n && lmul!(inv(norm(cov(ε)[k])), cov(ε)[k].data)

            # update Cholesky decomposition
            Cinv[k] = inv(cholesky(cov(ε)[k]).L)

            # update precision matrix
            Ω[k] = transpose(Cinv)[k] .* Cinv[k]

            # update scaling
            S[r][k] = Cinv[k] * U[r][k]
        end

        # dependent variable and regressor tensors
        Zr_scaled = tucker(Zr, Cinv)
        Xr = tucker(y_lag, S[r])

        # update loading
        loadings(A)[r] = dot(Zr_scaled, Xr) / norm(Xr)^2
    end

    return nothing
end
function als!(A::DynamicKruskal, ε::TensorNormal, y::AbstractArray, V::AbstractVector)
    dims = size(y)
    n = ndims(y) - 1

    # Cholesky decompositions of Σᵢ
    C = cholesky.(cov(ε))
    # inverse of Cholesky decompositions
    Cinv = inv.(getproperty.(C, :L))
    # precision matrices Ωᵢ
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = outer(A)

    # lag and lead variables
    y_lead = selectdim(y, n + 1, 2:last(dims))
    y_lag = selectdim(y, n + 1, 1:(last(dims) - 1))

    # Cholesky decomposition of V
    v_half = similar(loadings(A), rank(A) * (rank(A) + 1) ÷ 2, last(dims))
    for (t, Vt) in pairs(V)
        F = cholesky(Hermitian(Vt))
        offset = 0
        for r in 1:rank(A)
            for s in r:rank(A)
                v_half[offset + s - r + 1, t] .= F.L[s, r]
            end
            offset += rank(A) - r + 1
        end
    end

    # Gram matrix scaling
    scale = loadings(A) .^ 2
    for t in 1:last(dims), r in 1:rank(A), s in 1:r
        scale[r, t] += v_half[r + (s - 1) * rank(A), t]^2
    end

    # initialize regressor tensors
    X = [tucker(y_lag, U[r]) for r in 1:rank(A)]

    for r in 1:rank(A)
        r_ = setdiff(1:rank(A), r)
        for k in 1:n
            m = setdiff(1:n, k)
            # matricize dependent variable along k-th mode
            Zr = copy(y_lead)
            for (t, Zrt) in pairs(eachslice(Zr, dims = n + 1))
                Zrt .*= loadings(A)[r, t]
                for s in r_
                    τ = loadings(A)[r, t] .* loadings(A)[s, t]
                    for p in 1:min(r, s)
                        offset = (p - 1) * rank(A)
                        τ += v_half[r + offset, t] * v_half[s + offset, t]
                    end
                    Zrt .-= τ .* selectdim(X[s], n + 1, t)
                end
            end
            Zr_scaled = tucker(Zr, Cinv[m], m)
            Zkr = matricize(Zr_scaled, k)
            # matricize regressor along k-th mode
            Xr_scaled = tucker(X[r], Cinv[m])
            Xkr = matricize(Xr_scaled, k)

            # Gram matrix
            G = zeros(dims[k], dims[k])
            for (t, Xkrt) in pairs(eachslice(Xkr, dims = n + 1))
                G .+= scale[r, t] .* Xkrt * Xkrt'
            end
            # moment matrix
            M = Zkr * Xkr'

            # update factor k
            update_factor!(factors(A)[k][:, r], factors(A)[k + n][:, r], G \ M' * Ω[k],
                           inv(dot(factors(A)[k + n][:, r], Ω[k], factors(A)[k + n][:, r])))
            # update factor k+n
            update_factor!(factors(A)[k + n][:, r], factors(A)[k][:, r], M,
                           inv(dot(factors(A)[k][:, r], G, factors(A)[k][:, r])))

            # update outer product of Kruskal factors
            U[r][k] = factors(A)[k + n][:, r] * factors(A)[k][:, r]'
            # update regressor tensor
            X[r] = tucker(y_lag, U[r])
        end
    end

    # error tensor
    E = copy(y_lead)
    for (t, Et) in pairs(eachslice(E, dims = n + 1))
        for r in 1:rank(A)
            Et .-= loadings(A)[r, t] .* selectdim(X[r], n + 1, t)
        end
    end

    for k in 1:n
        m = setdiff(1:n, k)
        # matricize error along k-th mode
        E_scaled = tucker(E, Cinv[m], m)
        Ek = matricize(E_scaled, k)

        # update covariance
        mul!(cov(ε)[k].data, Ek, Ek')
        offset = 0
        for r in 1:rank(A)
            Xs = zero(X[r])
            for s in r:rank(A)
                Xs .+= v_half[offset + s - r + 1, t] .* X[s]
            end
            Xs_scaled = tucker(Xs, Cinv[m])
            Xks = matricize(Xs_scaled, k)
            mul!(cov(ε)[k].data, Xks, Xks', true, true)
            offset += rank(A) - r + 1
        end
        lmul!(inv((last(dims) - 1) * prod(dims[m])), cov(ε)[k].data)

        # normalize
        k != n && lmul!(inv(norm(cov(ε)[k])), cov(ε)[k].data)
    end

    return nothing
end

"""
    update_transition!(A, V, Γ)

Update transition dynamics of dynamic Kruskal coefficient tensor `A` using smoothed loadings
variance `V`, and autocovariance `Γ`.
"""
function update_transition!(A::DynamicKruskal, V::AbstractVector, Γ::AbstractVector)
    T = length(V)

    # objective closures
    function f_intercept(x, r)
        f = zero(x)
        for t in 2:T
            f -= 2 * (loadings(A)[r, t] - dynamics(A).diag[r] * loadings(A)[r, t - 1]) * x
        end
        f = x^2 + f / (T - 1)

        return f
    end
    function f_dynamics(x, r)
        scale = one(x) - x^2
        α = intercept(A)[r]
        f = zero(x)
        for t in 2:T
            f += V[t][r, r] + loadings(A)[r, t]^2 - 2 * α * loadings(A)[r, t]
            f += 2 *
                 (α * loadings(A)[r, t - 1] - Γ[t - 1][r, r] -
                  loadings(A)[r, t] * loadings(A)[r, t - 1]) * x
            f += (V[t - 1][r, r] + loadings(A)[r, t - 1]^2) * x^2
        end
        f = α^2 + f / (T - 1)

        return log(scale) + f * inv(scale)
    end

    # update dynamics
    for r in 1:rank(A)
        objective(x) = f_dynamics(x, r)
        res = optimize(f_dynamics, 0.0, 1.0)
        dynamics(A).diag[r] = Optim.minimizer(res)
    end
    for r in 1:rank(A)
        objective(x) = f_intercept(x, r)
        res = optimize(f_intercept, 0.0, 0.7 * (1 - dynamics(A).diag[r]))
        intercept(A)[r] = Optim.minimizer(res)
    end
    cov(A) .= I - dynamics(A) * dynamics(A)'

    return nothing
end

"""
    update_factor!(u, w, P, scale)

Update and normalize factor `u` using regression based on projection matrix `P`
and companion factor `w`.
"""
function update_factor!(u::AbstractVecOrMat, w::AbstractVecOrMat, P::AbstractMatrix,
                        scale::Real)
    # update factor
    mul!(u, P, w, scale, zero(eltype(u)))
    # normalize
    u .*= inv(norm(u))

    return nothing
end
