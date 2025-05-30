#=
utilities.jl

    Provides a collection of utility tools for working with tensor autoregressive models,
    such as moving average representation, orthogonalize responses, state space form for the
    dynamic model, as well as filter and smoother routines, simulation, and particle
    sampler.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/07/02
=#

"""
    confidence_bounds(model, periods, α, samples=100, burn=100, rng=Xoshiro())
        -> (lower, upper)

Compute Monte Carlo `α`% confidence bounds for impulse response functions of the static
tensor autoregressive model given by `model`. The confidence bounds are estimated using a
Monte Carlo simulation with `samples` and a burn-in period `burn`.
"""
function confidence_bounds(model::StaticTensorAutoregression, periods::Integer, α::Real;
                           samples::Integer = 100, burn::Integer = 100,
                           rng::AbstractRNG = Xoshiro())
    Ψ = Vector{Array}(undef, samples)

    # Monte Carlo estimation
    for s in 1:samples
        # simulate model
        sim = simulate(model, burn = burn, rng = rng)

        # fit simulated model
        fit!(sim)

        # moving average representation
        Ψ[s] = moving_average(sim, periods)

        # girf
        Ψ[s] = integrate(Ψ[s], cov(sim))
    end

    # quantiles
    lower_idx = round(Int, samples * α / 2)
    upper_idx = round(Int, samples * (1.0 - α / 2))

    # confidence bounds
    Ψ = cat(Ψ..., dims = ndims(Ψ[1]) + 1)
    Ψ_sorted = sort(Ψ, dims = ndims(Ψ))
    lower = selectdim(Ψ_sorted, ndims(Ψ_sorted), lower_idx)
    upper = selectdim(Ψ_sorted, ndims(Ψ_sorted), upper_idx)

    return (lower, upper)
end

"""
    moving_average(model, n) -> Ψ

Moving average, ``MA(∞)``, representation of the static tensor autoregressive model `model`,
computed up to the `n`th term.
"""
function moving_average(model::StaticTensorAutoregression, n::Integer)
    d = dims(model)
    nd = length(d)
    K = prod(d)
    Ty = eltype(data(model))

    # identity matrix
    Id = I(K)

    # matricize Kruskal tensor
    Am = matricize.(full.(coef(model)), Ref(1:nd))

    # moving average coefficients
    Ψm = zeros(Ty, K, K, n + 1)
    for (h, ψh) in pairs(eachslice(Ψm, dims = ndims(Ψm)))
        if h == 1
            ψh .= Id
        else
            for (p, Amp) in pairs(Am)
                if h > p
                    ψh .+= Ψm[:, :, h - p] * Amp
                else
                    break
                end
            end
        end
    end

    return stack(tensorize.(eachslice(Ψm, dims = ndims(Ψm)), Ref(1:nd), Ref((d..., d...))),
                 dims = 2nd + 1)
end

"""
    integrate(Ψ, Σ) -> Ψ_int

Integrate out all shocks except one from impulse responses `Ψ` using normality and
covariance matrix `Σ`.
"""
function integrate(Ψ::AbstractArray, Σ::Symmetric)
    n = ndims(Ψ) - 1
    m = (n ÷ 2 + 1):n

    # scaled covariance matrix
    σ = diag(Σ)
    Σ_scaled = Σ ./ transpose(σ)

    # isolate responses
    Ψ_int = similar(Ψ)
    for (h, ψ) in pairs(eachslice(Ψ, dims = n + 1))
        selectdim(Ψ_int, n + 1, h) .= tensorize(matricize(ψ, m) * Σ_scaled, m, size(ψ))
    end

    return Ψ_int
end
function integrate(Ψ::AbstractArray, Σ::AbstractVector)
    # scaled covariance matrix
    σ = diag.(Σ)
    Σ_scaled = broadcast.(/, Σ, transpose.(σ))

    # isolate responses
    Ψ_int = tucker(Ψ, Σ_scaled)

    return Ψ_int
end

"""
    companion(model[, t]) -> comp

Companion form of tensor autoregressive model `model`, where in case of a dynamic model the
companion form is obtained for time index `t`.
"""
function companion(model::StaticTensorAutoregression)
    d = dims(model)
    n = length(d)
    D = prod(d)
    K = lags(model) * D

    # companion matrix
    comp = zeros(K, K)
    # add identity matrix on off-diagonal
    idx = diagind(comp, D)
    comp[idx] .= 1.0
    # add coefficient tensors
    for (p, Ap) in pairs(coef(model))
        comp[1:D, ((p - 1) * D + 1):(p * D)] .= matricize(full(Ap), (n + 1):(2n))
    end

    return comp
end
function companion(model::DynamicTensorAutoregression, t::Integer)
    d = dims(model)
    n = length(d)
    D = prod(d)
    K = lags(model) * D

    # companion matrix
    comp = zeros(K, K)
    # add identity matrix on off-diagonal
    idx = diagind(comp, D)
    comp[idx] .= 1.0
    # add time varying coefficient tensors
    for (p, Ap) in pairs(coef(model))
        for (r, Apr) in pairs(full(Ap))
            comp[1:D, ((p - 1) * D + 1):(p * D)] .+= loadings(Ap)[r, t - p + 1] .*
                                                     matricize(Apr, (n + 1):(2n))
        end
    end

    return comp
end

"""
    lyaponov(model, m; samples = 250) -> γ

Lyaponov exponent for dynamic tensor autoregressive model `model` for length `m` and using
`samples` for the Monte Carlo estimate.
"""
function lyaponov(model::DynamicTensorAutoregression, m::Integer; samples::Integer = 250)
    γ = zeros(m)
    for _ in 1:samples
        # simulate from model
        sim = simulate(model)

        # companion form
        comp = [companion(sim, t) for t in (m + lags(model)):-1:(lags(model) + 1)]

        # Lyaponov exponents
        for i in 1:m
            γ[i] += log(opnorm(prod(first(comp, i))))
        end
    end

    return γ ./ samples
end

"""
    sampler(model, samples, periods, conditional[, shock, index], rng) -> paths

Sample conditional paths from the model `periods` ahead conditional on `conditional` given
an optional `shock` at series given by `index` using random number generator `rng`.
"""
function sampler(model::DynamicTensorAutoregression, samples::Integer, periods::Integer,
                 conditional::Integer, rng::AbstractRNG)
    # sample particles
    particles = particle_sampler(model, periods, conditional, samples, rng)

    # sample observation noise
    noise = simulate(dist(model), samples * periods, rng)

    # sample paths
    paths = similar(data(model), dims(model)..., periods, samples)
    path_sampler!(paths, model, particles, noise, conditional)

    return paths
end
function sampler(model::DynamicTensorAutoregression, samples::Integer, periods::Integer,
                 conditional::Integer, shock::Real, index::Dims, rng::AbstractRNG)
    # sample particles
    particles = particle_sampler(model, periods, conditional, samples, rng)

    # sample observation noise
    noise = simulate(dist(model), samples * periods, rng)
    # shock
    noise[index..., 1] = shock

    # sample paths
    paths = similar(data(model), dims(model)..., periods, samples)
    path_sampler!(paths, model, particles, noise, conditional)

    return paths
end
function sampler(model::DynamicTensorAutoregression, samples::Integer, periods::Integer,
                 conditional::AbstractUnitRange, rng::AbstractRNG)
    d = dims(model)
    n = length(d)

    # sample particles
    particles = particle_sampler(model, periods, conditional, samples, rng)

    # sample observation noise
    noise = simulate(dist(model), periods * samples * length(conditional), rng)

    # sample paths
    paths = similar(data(model), d..., periods, length(conditional), samples)
    for (t, conditional_paths) in pairs(eachslice(paths, dims = ndims(paths) - 1))
        path_sampler!(conditional_paths, model,
                      selectdim(particles, ndims(particles) - 1, t),
                      selectdim(noise, n + 1,
                                ((t - 1) * samples * periods + 1):(t * samples * periods)),
                      conditional[t])
    end

    return paths
end
function sampler(model::DynamicTensorAutoregression, samples::Integer, periods::Integer,
                 conditional::AbstractUnitRange, shock::Real, index::Dims, rng::AbstractRNG)
    d = dims(model)
    n = length(d)

    # sample particles
    particles = particle_sampler(model, periods, conditional, samples, rng)

    # sample observation noise
    noise = simulate(dist(model), periods * samples * length(conditional), rng)
    # shock
    noise[index..., 1:(periods * samples):end] .= shock

    # sample paths
    paths = similar(data(model), d..., periods, length(conditional), samples)
    for (t, conditional_paths) in pairs(eachslice(paths, dims = ndims(paths) - 1))
        path_sampler!(conditional_paths, model,
                      selectdim(particles, ndims(particles) - 1, t),
                      selectdim(noise, n + 1,
                                ((t - 1) * samples * periods + 1):(t * samples * periods)),
                      conditional[t])
    end

    return paths
end

"""
    path_sampler!(paths, model, particles, noise, conditional)

Sample conditional paths from the `model` given simulated `particles` and `noise`.
"""
function path_sampler!(paths::AbstractArray, model::DynamicTensorAutoregression,
                       particles::AbstractArray, noise::AbstractArray, conditional::Integer)
    n = length(dims(model))
    Rc = cumsum(rank(model))

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # sample paths
    for (s, path) in pairs(eachslice(paths, dims = ndims(paths)))
        for (h, element) in pairs(eachslice(path, dims = ndims(path)))
            # sample element
            element .= selectdim(noise, n + 1, (s - 1) * size(path, ndims(path)) + h)
            for (p, Rp) in pairs(rank(model))
                # lag
                if h <= p
                    lag = selectdim(data(model), n + 1, conditional + h - p)
                else
                    lag = selectdim(path, ndims(path), h - p)
                end
                # propagate
                for r in 1:Rp
                    i = (p > 1 ? Rc[p - 1] : 0) + r
                    element .+= particles[i, h, s] .* tucker(lag, U[p][r])
                end
            end
        end
    end

    return nothing
end

"""
    particle_sampler(model, periods, conditional, samples, rng) -> particles

Forward particle sampler of the filtered state ``αₜ₊ₕ`` for the dynamic tensor
autoregressive model `model`, with the number of forward periods given by `periods`
conditional on `conditional` and using random number generator `rng`.
"""
function particle_sampler(model::DynamicTensorAutoregression, periods::Integer,
                          conditional::Integer, samples::Integer,
                          rng::AbstractRNG)
    # state transition parameters
    (c, T, Q) = state_transition_params(model)
    # filter
    (a, P, _, _) = filter(model, predict = true)
    a_hat = a[conditional - lags(model) + 1]
    P_hat = P[conditional - lags(model) + 1]

    # particle sampling
    particles = similar(a_hat, length(a_hat), periods, samples)
    base_dist = MvNormal(a_hat, Symmetric(P_hat))
    for s in 1:samples, h in 1:periods
        if h == 1
            particles[:, h, s] = rand(rng, base_dist)
        else
            particles[:, h, s] = rand(rng, MvNormal(c + T * particles[:, h - 1, s], Q))
        end
    end

    return particles
end
function particle_sampler(model::DynamicTensorAutoregression, periods::Integer,
                          conditional::AbstractUnitRange, samples::Integer,
                          rng::AbstractRNG)
    # state transition parameters
    (c, T, Q) = state_transition_params(model)
    # filter
    (a, P, _, _) = filter(model, predict = true)

    # particle sampling
    particles = similar(a[1], length(a[1]), periods, length(conditional), samples)
    for s in 1:samples, (t, time) in pairs(conditional), h in 1:periods
        if h == 1
            particles[:, h, t, s] = rand(rng,
                                         MvNormal(a[time - lags(model) + 1],
                                                  Symmetric(P[time - lags(model) + 1])))
        else
            particles[:, h, t, s] = rand(rng,
                                         MvNormal(c + T * particles[:, h - 1, t, s], Q))
        end
    end

    return particles
end

"""
    state_transition_params(model) -> (c, T, Q)

State transition parameters ``c``, ``T``, and ``Q`` of the state space form of the dynamic
tensor autoregressive model `model`.
"""
function state_transition_params(model::DynamicTensorAutoregression)
    c = vcat(intercept.(coef(model))...)
    T = Diagonal(vcat(getfield.(dynamics.(coef(model)), :diag)...))
    Q = Diagonal(vcat(getfield.(cov.(coef(model)), :diag)...))

    return (c, T, Q)
end

"""
    state_space_init(model) -> (a1, P1)

State space initial conditions of the state space form of the dynamic tensor autoregressive
model `model`.
"""
function state_space_init(model::DynamicTensorAutoregression)
    T = eltype(data(model))
    R = sum(rank(model))
    a1 = zeros(T, R)
    P1 = Matrix{T}(I, R, R)

    return (a1, P1)
end

"""
    loading_matrix(model) -> L

High-dimensional loading matrix for the state space form of the dynamic tensor
autoregressive model `model`.
"""
function loading_matrix(model::DynamicTensorAutoregression)
    n = length(dims(model))

    # outer product of Kruskal factors
    U = outer.(coef(model))

    # high-dimensional time-varying loading matrix
    Z = [[tucker(data(model), U[p][r]) for r in 1:Rp] for (p, Rp) in pairs(rank(model))]
    L_unstacked = stack([[stack([vec(selectdim(Zpr, n + 1, t)) for Zpr in Zp])
                          for t in (lags(model) - p + 1):(nobs(model) - p)]
                         for (p, Zp) in pairs(Z)])

    return broadcast(splat(hcat), eachrow(L_unstacked))
end

"""
    collapse(model; objective = false) -> (y_low, Z_low, H_low[, M])

Collapsed system components for the collapsed state space form of the dynamic tensor
autoregressive model `model` following the approach of Jungbacker and Koopman (2015).
Optional `objective` boolean indicating whether the collapsed system components are used for
objective function (log-likelihood) computation, in which case additionally the annihilator
matrix is returned.
"""
function collapse(model::DynamicTensorAutoregression; objective::Bool = false)
    n = length(dims(model))

    # high-dimensional time-varying loading matrix
    Z = loading_matrix(model)
    # linearly independent columns
    F = qr.(Z)
    ic = [broadcast(!iszero, eachcol(Ft.R)) for Ft in F]

    # concentration matrix
    Ω = concentration(model, full = true)

    # collapsing matrices
    Z_basis = [Zt[:, ic[t]] for (t, Zt) in pairs(Z)]
    A_low = transpose.(Z_basis) .* Ref(Ω)

    # collapsed system
    y_low = [A_low[t] * vec(yt)
             for (t, yt) in enumerate(Iterators.drop(eachslice(data(model), dims = n + 1),
                                                     lags(model)))]
    Z_low = A_low .* Z
    H_low = A_low .* Z_basis

    # annihilator matrix for log-likelihood
    if objective
        M = Ref(I) .- Z_basis .* (H_low .\ A_low)

        return (y_low, Z_low, H_low, M)
    else
        return (y_low, Z_low, H_low)
    end
end

"""
    filter(model; predict = false) -> (a, P, v, F)

Collapsed Kalman filter for the dynamic tensor autoregressive model `model`. Returns the
filtered state `a`, covariance `P`, forecast error `v`, and forecast error variance `F`. If
`predict` is `true` the filter reports the one-step ahead out-of-sample prediction.
"""
function filter(model::DynamicTensorAutoregression; predict::Bool = false)
    # collapsed state space system
    (y, Z, H) = collapse(model)
    (c, T, Q) = state_transition_params(model)
    (a1, P1) = state_space_init(model)

    # initialize filter output
    n = length(y) + (predict ? 1 : 0)
    a = similar(y, typeof(a1), n)
    P = similar(y, typeof(P1), n)
    v = similar(y)
    F = similar(y, typeof(P1))

    # initialize storage
    ZtFinv = similar(P1)
    att = similar(a1)
    Ptt = similar(P1)

    # initialize filter
    a[1] = a1
    P[1] = P1

    # filter
    for t in eachindex(y)
        # forecast error
        v[t] = y[t] - Z[t] * a[t]
        F[t] = Z[t] * P[t] * Z[t]' + H[t]

        # update
        ZtFinv .= (F[t] \ Z[t])'
        att .= a[t] + P[t] * ZtFinv * v[t]
        Ptt .= P[t] - P[t] * ZtFinv * Z[t] * P[t]

        # prediction
        if predict || t < length(y)
            a[t + 1] = T * att + c
            P[t + 1] = T * Ptt * T' + Q
            # enforce symmetry for numerical stability
            P[t + 1] = 0.5 * (P[t + 1] + P[t + 1]')
        end
    end

    return (a, P, v, F)
end

"""
    _filter_smoother(y, Z, H, c, T, Q, a1, P1) -> (a, P, v, ZtFinv)

Collapsed Kalman filter for the dynamic tensor autoregressive model used internally by the
`smoother` routine to avoid duplicate expensive computation of state space system matrices.
"""
function _filter_smoother(y, Z, H, c, T, Q, a1, P1)
    # initialize filter output
    a = similar(y, typeof(a1))
    P = similar(y, typeof(P1))
    v = similar(y)
    ZtFinv = similar(y, typeof(P1))

    # initialize storage
    F = similar(P1)
    att = similar(a1)
    Ptt = similar(P1)

    # initialize filter
    a[1] = a1
    P[1] = P1

    # filter
    for t in eachindex(y)
        # forecast error
        v[t] = y[t] - Z[t] * a[t]
        F .= Z[t] * P[t] * Z[t]' + H[t]

        # update
        ZtFinv[t] = (F \ Z[t])'
        att .= a[t] + P[t] * ZtFinv[t] * v[t]
        Ptt .= P[t] - P[t] * ZtFinv[t] * Z[t] * P[t]

        # prediction
        if t < length(y)
            a[t + 1] = T * att + c
            P[t + 1] = T * Ptt * T' + Q
            # enforce symmetry for numerical stability
            P[t + 1] = 0.5 * (P[t + 1] + P[t + 1]')
        end
    end

    return (a, P, v, ZtFinv)
end

"""
    _filter_likelihood(y, Z, H, c, T, Q, a1, P1) -> (v, F)

Collapsed Kalman filter for the dynamic tensor autoregressive model used internally by the
`loglikelihood` routine to avoid duplicate expensive computation of collapsing components
and state space system matrices.
"""
function _filter_likelihood(y, Z, H, c, T, Q, a1, P1)
    # initialize filter output
    v = similar(y)
    F = similar(y, typeof(P1))

    # initialize storage
    ZtFinv = similar(P1)
    att = similar(a1)
    Ptt = similar(P1)

    # initialize filter
    a = copy(a1)
    P = copy(P1)

    # filter
    for t in eachindex(y)
        # forecast error
        v[t] = y[t] - Z[t] * a
        F[t] = Z[t] * P * Z[t]' + H[t]

        # update
        ZtFinv .= (F[t] \ Z[t])'
        att .= a + P * ZtFinv * v[t]
        Ptt .= P - P * ZtFinv * Z[t] * P

        # prediction
        if t < length(y)
            a .= T * att + c
            P = T * Ptt * T' + Q
            # enforce symmetry for numerical stability
            P = 0.5 * (P + P')
        end
    end

    return (v, F)
end

"""
    smoother(model) -> (α, V, Γ)

Collapsed Kalman smoother for the dynamic tensor autoregressive model `model`. Returns the
smoothed state `α`, covariance `V`, and autocovariance `Γ`.
"""
function smoother(model::DynamicTensorAutoregression)
    # collapsed state space system
    (y, Z, H) = collapse(model)
    (c, T, Q) = state_transition_params(model)
    (a1, P1) = state_space_init(model)

    # filter
    (a, P, v, ZtFinv) = _filter_smoother(y, Z, H, c, T, Q, a1, P1)

    # initialize smoother output
    α = similar(a)
    V = similar(P)
    Γ = similar(P, length(y) - 1)

    # initialize smoother
    r = zero(a[1])
    N = zero(P[1])
    L = similar(P[1])

    # smoother
    for t in reverse(eachindex(y))
        L .= T - T * P[t] * ZtFinv[t] * Z[t]

        # backward recursion
        r .= ZtFinv[t] * v[t] + L' * r
        N .= ZtFinv[t] * Z[t] + L' * N * L

        # smoothing
        α[t] = a[t] + P[t] * r
        V[t] = P[t] - P[t] * N * P[t]
        t > 1 && (Γ[t - 1] = I - P[t] * N)
        t < length(y) && (Γ[t] *= L * P[t])
    end

    return (α, V, Γ)
end

"""
    simulate(A, S, rng) -> λ

Simulate the dynamic loadings from the dynamic Kruskal coefficient tensor `A` `S` times
using the random number generator `rng`.
"""
function simulate(A::DynamicKruskal, S::Integer, rng::AbstractRNG)
    λ = similar(loadings(A), rank(A), S)
    dist = MvNormal(cov(A))
    # simulate
    for (s, λs) in pairs(eachcol(λ))
        if s == 1
            # initial condition
            λs .= rand(rng, dist)
        else
            λs .= intercept(A) + dynamics(A) * λ[:, s - 1] + rand(rng, dist)
        end
    end

    return λ
end

"""
    simulate(ε, S, rng) -> e

Simulate from the tensor error distribution `ε` `S` times using the random number generator
`rng`.
"""
simulate(ε::WhiteNoise, S::Integer, rng::AbstractRNG) = error("simulating data from white noise error not supported.")
function simulate(ε::TensorNormal, S::Integer, rng::AbstractRNG)
    d = size.(cov(ε), 1)

    # Cholesky decompositions of Σᵢ
    C = getproperty.(cholesky.(cov(ε)), :L)

    return tucker(randn(rng, d..., S), C)
end
