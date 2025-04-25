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
function update!(model::StaticTensorAutoregression{<:AbstractArray, <:WhiteNoise,
                                                   <:AbstractVector})
    n = length(dims(model))

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # lag and lead variables
    y_lead = selectdim(data(model), n + 1, (lags(model) + 1):nobs(model))
    y_lags = [selectdim(data(model), n + 1, (lags(model) - p + 1):(nobs(model) - p))
              for p in 1:lags(model)]

    # initialize dependent variable tensor
    Zpr = similar(y_lead)
    # initialize residuals
    resid = copy(y_lead)

    for (p, Rp) in pairs(rank(model))
        for r in 1:Rp
            # dependent variable tensor
            Zpr .= y_lead
            for (q, Rq) in pairs(rank(model))
                r_ = q == p ? setdiff(1:Rq, r) : 1:Rq
                for s in r_
                    Zpr .-= loadings(model)[q][s] .* tucker(y_lags[q], U[q][s])
                end
            end
            for k in 1:n
                k_ = setdiff(1:n, k)
                # matricize dependent variable along k-th mode
                Zkpr = matricize(Zpr, k)
                # matricize regressor along k-th mode
                Xpr = tucker(y_lags[p], U[p][r][k_], k_)
                Xkpr = matricize(Xpr, k)

                # Gram matrix
                G = Xkpr * Xkpr'
                # moment matrix
                M = Zkpr * Xkpr'

                # update factor k
                @views update_factor!(factors(model)[p][k][:, r],
                                      factors(model)[p][k + n][:, r],
                                      G \ M',
                                      inv(loadings(model)[p][r]))
                # update factor k+n
                @views update_factor!(factors(model)[p][k + n][:, r],
                                      factors(model)[p][k][:, r],
                                      M,
                                      inv(loadings(model)[p][r] *
                                          dot(factors(model)[p][k][:, r], G,
                                              factors(model)[p][k][:, r])))

                # update outer product of Kruskal factors
                U[p][r][k] .= factors(model)[p][k + n][:, r] * factors(model)[p][k][:, r]'
            end

            # regressor tensor
            Xpr = tucker(y_lags[p], U[p][r])

            # update loading
            loadings(model)[p][r] = dot(Zpr, Xpr) / norm(Xpr)^2

            # update residuals
            resid .-= loadings(model)[p][r] .* Xpr
        end
    end

    # update covariance
    E = matricize(resid, 1:n)
    mul!(cov(ε).data, E, E')

    return nothing
end
function update!(model::StaticTensorAutoregression{<:AbstractArray, <:TensorNormal,
                                                   <:AbstractVector})
    d = dims(model)
    n = length(d)

    # Cholesky decompositions of Σᵢ
    C = cholesky.(cov(model))
    # inverse of Cholesky decompositions
    Cinv = inv.(getproperty.(C, :L))
    # precision matrices Ωᵢ
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # scaling
    S = [[[Cinv[i] * U[p][r][i] for i in 1:n] for r in 1:Rp]
         for (p, Rp) in pairs(rank(model))]

    # lag and lead variables
    y_lead = selectdim(data(model), n + 1, (lags(model) + 1):nobs(model))
    y_lags = [selectdim(data(model), n + 1, (lags(model) - p + 1):(nobs(model) - p))
              for p in 1:lags(model)]

    # initialize dependent variable tensor
    Zpr = similar(y_lead)

    for (p, Rp) in pairs(rank(model))
        for r in 1:Rp
            # dependent variable tensor
            Zpr .= y_lead
            for (q, Rq) in pairs(rank(model))
                r_ = q == p ? setdiff(1:Rq, r) : 1:Rq
                for s in r_
                    Zpr .-= loadings(model)[q][s] .* tucker(y_lags[q], U[q][s])
                end
            end
            for k in 1:n
                k_ = setdiff(1:n, k)
                # matricize dependent variable along k-th mode
                Zpr_scaled = tucker(Zpr, Cinv[k_], k_)
                Zkpr = matricize(Zpr_scaled, k)
                # matricize regressor along k-th mode
                Xpr = tucker(y_lags[p], S[p][r][k_], k_)
                Xkpr = matricize(Xpr, k)

                # Gram matrix
                G = Xkpr * Xkpr'
                # moment matrix
                M = Zkpr * Xkpr'

                # update factor k
                @views update_factor!(factors(model)[p][k][:, r],
                                      factors(model)[p][k + n][:, r],
                                      G \ M' * Ω[k],
                                      inv(loadings(model)[p][r] *
                                          dot(factors(model)[p][k + n][:, r], Ω[k],
                                              factors(model)[p][k + n][:, r])))
                # update factor k+n
                @views update_factor!(factors(model)[p][k + n][:, r],
                                      factors(model)[p][k][:, r],
                                      M,
                                      inv(loadings(model)[p][r] *
                                          dot(factors(model)[p][k][:, r], G,
                                              factors(model)[p][k][:, r])))

                # update outer product of Kruskal factors
                U[p][r][k] .= factors(model)[p][k + n][:, r] * factors(model)[p][k][:, r]'

                # update scaling
                mul!(S[p][r][k], Cinv[k], U[p][r][k])
            end

            # dependent variable and regressor tensors
            Zpr_scaled = tucker(Zpr, Cinv)
            Xpr = tucker(y_lags[p], S[p][r])

            # update loading
            loadings(model)[p][r] = dot(Zpr_scaled, Xpr) / norm(Xpr)^2
        end
    end

    # error tensor
    E = copy(y_lead)
    for (p, Rp) in pairs(rank(model))
        for r in 1:Rp
            E .-= loadings(model)[p][r] .* tucker(y_lags[p], U[p][r])
        end
    end

    for k in 1:n
        k_ = setdiff(1:n, k)
        # matricize error along k-th mode
        E_scaled = tucker(E, Cinv[k_], k_)
        Ek = matricize(E_scaled, k)

        # update covariance
        mul!(cov(model)[k].data, Ek, Ek',
             inv((nobs(model) - lags(model)) * prod(d[k_])), false)

        # normalize
        k != n && lmul!(inv(norm(cov(model)[k])), cov(model)[k].data)

        # update inverse of Cholesky decomposition
        Cinv[k] .= inv(getproperty(cholesky(cov(model)[k]), :L))
    end

    return nothing
end
function update!(model::DynamicTensorAutoregression)
    # E-step
    # smoother
    (α, V, Γ) = smoother(model)
    R = sum(rank(model))
    Rc = cumsum(rank(model))
    for (t, αt) in pairs(α)
        for j in 1:R
            p = sum(x -> isless(x, j), Rc) + 1
            r = j - get(Rc, p - 1, 0)
            loadings(model)[p][r, t] = αt[j]
        end
    end

    # M-step
    update_transition_params!(model, V, Γ)
    update_obs_params!(model, V)

    return nothing
end

"""
    update_obs_params!(model, V)

Update observation equation parameters of the dynamic tensor autoregressive model `model`
based on smoothed loading variance `V` and using an alternating least squares (ALS) solve.
"""
function update_obs_params!(model::DynamicTensorAutoregression, V::AbstractVector)
    d = dims(model)
    n = length(d)
    R = sum(rank(model))
    Rc = cumsum(rank(model))

    # Cholesky decompositions of Σᵢ
    C = cholesky.(cov(model))
    # inverse of Cholesky decompositions
    Cinv = inv.(getproperty.(C, :L))
    # precision matrices Ωᵢ
    Ω = transpose.(Cinv) .* Cinv

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # scaling
    S = [[[Cinv[i] * U[p][r][i] for i in 1:n] for r in 1:Rp]
         for (p, Rp) in pairs(rank(model))]

    # lag and lead variables
    y_lead = selectdim(data(model), n + 1, (lags(model) + 1):nobs(model))
    y_lags = [selectdim(data(model), n + 1, (lags(model) - p + 1):(nobs(model) - p))
              for p in 1:lags(model)]

    # Cholesky decomposition of V
    v_half = similar(V[1], R * (R + 1) ÷ 2, length(V))
    for (t, Vt) in pairs(V)
        F = cholesky(Hermitian(Vt))
        offset = 0
        for r in 1:R
            for s in r:R
                v_half[s - r + 1 + offset, t] = F.L[s, r]
            end
            offset += R - r + 1
        end
    end

    # Gram matrix scaling
    scale = vcat(loadings(model)...) .^ 2
    for t in 1:(nobs(model) - lags(model))
        for r in 1:R
            offset = 0
            for s in 1:r
                scale[r, t] += v_half[r + offset, t]^2
                offset += R - s
            end
        end
    end

    # initialize regressor tensors
    X = [[tucker(y_lags[p], U[p][r]) for r in 1:Rp] for (p, Rp) in pairs(rank(model))]
    # initialize dependent variable tensor
    Zpr = similar(y_lead)

    for (p, Rp) in pairs(rank(model))
        for r in 1:Rp
            # dependent variable tensor
            Zpr .= y_lead
            for (t, Zprt) in pairs(eachslice(Zpr, dims = n + 1))
                Zprt .*= loadings(model)[p][r, t]
                for (q, Rq) in pairs(rank(model))
                    r_ = q == p ? setdiff(1:Rq, r) : 1:Rq
                    for s in r_
                        τ = loadings(model)[p][r, t] .* loadings(model)[q][s, t]
                        offset = 0
                        for j in 1:min(r, s)
                            τ += v_half[r + offset, t] * v_half[s + offset, t]
                            offset += R - j
                        end
                        Zprt .-= τ .* selectdim(X[q][s], n + 1, t)
                    end
                end
            end
            for k in 1:n
                k_ = setdiff(1:n, k)
                # matricize dependent variable along k-th mode
                Zpr_scaled = tucker(Zpr, Cinv[k_], k_)
                Zkpr = matricize(Zpr_scaled, k)
                # matricize regressor along k-th mode
                Xpr_scaled = tucker(y_lags[p], S[p][r][k_], k_)
                Xkpr = matricize(Xpr_scaled, k)

                # Gram matrix
                G = zeros(d[k], d[k])
                N_ = prod(d[k_])
                for t in 1:(nobs(model) - lags(model))
                    Xkprt = view(Xkpr, :, (N_ * (t - 1) + 1):(N_ * t))
                    mul!(G, Xkprt, Xkprt', scale[r, t], true)
                end
                # moment matrix
                M = Zkpr * Xkpr'

                # update factor k
                @views update_factor!(factors(model)[p][k][:, r],
                                      factors(model)[p][k + n][:, r],
                                      G \ M' * Ω[k],
                                      inv(dot(factors(model)[p][k + n][:, r], Ω[k],
                                              factors(model)[p][k + n][:, r])))
                # update factor k+n
                @views update_factor!(factors(model)[p][k + n][:, r],
                                      factors(model)[p][k][:, r],
                                      M,
                                      inv(dot(factors(model)[p][k][:, r], G,
                                              factors(model)[p][k][:, r])))

                # update outer product of Kruskal factors
                U[p][r][k] .= factors(model)[p][k + n][:, r] * factors(model)[p][k][:, r]'
                # update scaling
                mul!(S[p][r][k], Cinv[k], U[p][r][k])
            end
            # update regressor tensor
            X[p][r] .= tucker(y_lags[p], U[p][r])
        end
    end

    # error tensor
    E = copy(y_lead)
    for (t, Et) in pairs(eachslice(E, dims = n + 1))
        for (p, Rp) in pairs(rank(model))
            for r in 1:Rp
                Et .-= loadings(model)[p][r, t] .* selectdim(X[p][r], n + 1, t)
            end
        end
    end

    # accumulate scaled regressor tensors
    offset = 0
    Xs = [zero(y_lead) for r in 1:R]
    for j in 1:R
        p = sum(x -> isless(x, j), Rc) + 1
        r = j - get(Rc, p - 1, 0)
        for l in j:R
            q = sum(x -> isless(x, l), Rc) + 1
            s = l - get(Rc, q - 1, 0)
            for (t, Xsjt) in pairs(eachslice(Xs[j], dims = n + 1))
                Xsjt .+= v_half[l - j + 1 + offset, t] .* selectdim(X[q][s], n + 1, t)
            end
        end
        offset += R - j + 1
    end

    for k in 1:n
        k_ = setdiff(1:n, k)
        # matricize error along k-th mode
        E_scaled = tucker(E, Cinv[k_], k_)
        Ek = matricize(E_scaled, k)

        # update covariance
        mul!(cov(model)[k].data, Ek, Ek')
        offset = 0
        for Xsr in Xs
            Xsr_scaled = tucker(Xsr, Cinv[k_], k_)
            Xskr = matricize(Xsr_scaled, k)
            mul!(cov(model)[k].data, Xskr, Xskr', true, true)
        end
        lmul!(inv((nobs(model) - lags(model)) * prod(d[k_])), cov(model)[k].data)

        # normalize
        k != n && lmul!(inv(norm(cov(model)[k])), cov(model)[k].data)

        # update inverse of Cholesky decomposition
        Cinv[k] .= inv(getproperty(cholesky(cov(model)[k]), :L))
    end

    return nothing
end

"""
    update_transition_params!(model, V, Γ)

Update transition dynamics of the dynamic Kruskal tensor of the dynamic tensor autoregresive
model `model` using smoothed loadings variance `V`, and autocovariance `Γ`.
"""
function update_transition_params!(model::DynamicTensorAutoregression, V::AbstractVector,
                                   Γ::AbstractVector)
    T = length(V)
    Rc = cumsum(rank(model))

    # update transition dynamics
    for (p, Ap) in pairs(coef(model))
        for r in 1:rank(Ap)
            s = (p > 1 ? Rc[p - 1] : 0) + r

            # objective function
            function objective(x)
                αr, ϕr, σr = x[1], 2.0 / (1.0 + exp(-x[2])) - 1.0, exp(x[3])
                e = 0.0
                for t in 2:T
                    μ2 = V[t][s, s] + loadings(Ap)[r, t]^2
                    μ2_lag = V[t - 1][s, s] + loadings(Ap)[r, t - 1]^2
                    μ_cross = Γ[t - 1][s, s] + loadings(Ap)[r, t] * loadings(Ap)[r, t - 1]
                    e += μ2 - 2 * αr * loadings(Ap)[r, t] +
                         2 * ϕr * (αr * loadings(Ap)[r, t - 1] - μ_cross) + αr^2 +
                         ϕr^2 * μ2_lag
                end

                return log(σr) + e / (σr * (T - 1))
            end

            # optimize
            x0 = [
                intercept(Ap)[r],
                log((dynamics(Ap).diag[r] + 1.0) / (2.0 * (1.0 - dynamics(Ap).diag[r]))),
                log(cov(Ap).diag[r])
            ]
            res = optimize(objective, x0, LBFGS(), autodiff = :forward)

            # store parameters
            intercept(Ap)[r] = Optim.minimizer(res)[1]
            dynamics(Ap).diag[r] = 2.0 / (1.0 + exp(-Optim.minimizer(res)[2])) - 1.0
            cov(Ap).diag[r] = exp(Optim.minimizer(res)[3])
        end
    end

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
