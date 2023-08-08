#=
utilities.jl

    Provides a collection of utility tools for working with tensor 
    autoregressive models, such as moving average representation, orthogonalize 
    responses, state space form for the dynamic model, as well as filter and 
    smoother routines, simulation, and particle sampler. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/07/02
=#

"""
    confidence_bounds(
        model, 
        periods, 
        α, 
        orth, 
        samples=100, 
        burn=100, 
        rng=Xoshiro()
    ) -> (lower, upper)

Compute Monte Carlo `α`% confidence bounds for impulse response functions of the
tensor autoregressive model given by `model`. The confidence bounds are
estimated using a Monte Carlo simulation with `samples` and a burn-in period
`burn`.
"""
function confidence_bounds(
    model::AbstractTensorAutoregression, 
    periods::Integer,
    α::Real, 
    orth::Bool, 
    samples::Integer=100, 
    burn::Integer=100, 
    rng::AbstractRNG=Xoshiro()
)   
    Ψ = Vector{Array}(undef, samples)

    # Monte Carlo estimation
    for s = 1:samples
        # simulate model
        sim = simulate(model, burn=burn, rng=rng)

        # fit simulated model
        fit!(sim)

        # moving average representation
        Ψ[s] = moving_average(sim, periods)

        # orthogonalize
        orth ? Ψ[s] = orthogonalize(Ψ[s], cov(sim)) : nothing
    end

    # quantiles
    lower_idx = round(Int, samples * α / 2)
    upper_idx = round(Int, samples * (1. - α / 2))

    # confidence bounds
    Ψ = cat(Ψ..., dims=ndims(Ψ[1])+1)
    Ψ_sorted = sort(Ψ, dims=ndims(Ψ))
    lower = selectdim(Ψ_sorted, ndims(Ψ_sorted), lower_idx)
    upper = selectdim(Ψ_sorted, ndims(Ψ_sorted), upper_idx)

    return (lower, upper)
end

"""
    moving_average(model, n) -> Ψ

Moving average, ``MA(∞)``, representation of the tensor autoregressive model
`model`, computed up to the `n`th term.
"""
function moving_average(model::StaticTensorAutoregression, n::Integer)
    dims = size(coef(model))
    R = length(dims)÷2+1:length(dims)

    # matricize Kruskal tensor
    An = matricize(full(coef(model)), R)

    # moving average coefficients
    Ψ = zeros(dims..., n+1)
    for (h, ψh) ∈ pairs(eachslice(Ψ, dims=ndims(Ψ)))
        ψh .= tensorize(An^(h-1), R, dims)
    end

    return Ψ
end

function moving_average(model::DynamicTensorAutoregression, n::Integer)
    dims = size(coef(model))
    R = length(dims)÷2+1:length(dims)-1

    # tensorize identity matrix
    Id = tensorize(I(prod(R)), R, dims[1:end-1])

    # matricize Kruskal tensor
    An = matricize(full(coef(model)), R)

    # moving average coefficients
    Ψ = zeros(dims[1:end-1]..., n+1, last(dims))
    for (t, ψt) ∈ pairs(eachslice(Ψ, dims=ndims(Ψ)))
        # sample particles
        particles = get_particles(selectdim(y, ndims(y), 1:t+1), A, ε, n)
        # cumulative product
        Λ = cumprod(particles, dims=ndims(particles))
        # uncertainty aggregation
        λ = dropdims(mean(Λ, dims=2), dims=2)
        for (h, ψh) ∈ pairs(eachslice(ψt, dims=ndims(ψt)))
            if h == 1
                ψh .= Id 
            else
                ψh .= λ[1,h-1] * tensorize(An^(h-1), R, size(ψh))
            end
        end
    end

    return Ψ
end

"""
    orthogonalize(Ψ, Σ) -> Ψ_orth

Orthogonalize impulse responses `Ψ` using the Cholesky decomposition of
covariance matrix `Σ`.
"""
function orthogonalize(Ψ::AbstractArray, Σ::AbstractMatrix)
    n = ndims(Ψ) - 1
    R = n÷2+1:n

    # Cholesky decomposition of Σ
    C = cholesky(Hermitian(Σ))

    # orthogonalize responses
    Ψ_orth = similar(Ψ)
    for (h, ψ) ∈ pairs(eachslice(Ψ, dims=n+1))
        selectdim(Ψ_orth, n+1, h) .= tensorize(matricize(ψ, R) * C.L, R, size(ψ))
    end

    return Ψ_orth
end

function orthogonalize(Ψ::AbstractArray, Σ::AbstractVector)
    # Cholesky decompositions of Σᵢ
    C = [cholesky(Hermitian(Σi)).U for Σi ∈ Σ]

    # orthogonalize responses
    Ψ_orth = tucker(Ψ, C)

    return Ψ_orth
end

"""
    get_particles(y, A, ε, periods) -> particles

Helper function for retrieving forward sampled particles for the dynamic model.
"""
function get_particles(y::AbstractArray, A::DynamicKruskal, ε::TensorNormal, periods::Integer)
    # collapsed state space system
    (y_star, Z_star, a1, P1) = state_space(y, A, ε)
    # filter
    (a, P, v, _, K) = filter(
        y_star, 
        Z_star, 
        intercept(A), 
        dynamics(A), 
        cov(A), 
        a1, 
        P1
    )
    # predict
    â = dynamics(A) * a[end] + K[end] * v[end] + intercept(A)
    P̂ = dynamics(A) * P[end] * (dynamics(A) - K[end] * Z_star[end])' + cov(A)

    # sample particles
    particles = particle_sampler(
        â, 
        P̂, 
        intercept(A), 
        dynamics(A), 
        cov(A), 
        periods, 
        1000, 
        Xoshiro()
    )

    return particles
end

"""
    particle_sampler(a, P, c, T, Q, periods, samples, rng) -> particles

Forward particle sampler of the filtered state `a` with corresponding variance
`P`, state equation system matrices `T` and `Q`, and state mean adjustment `c`
with the number of forward periods given by `periods`, using random number
generator `rng`.
"""
function particle_sampler(
    a::AbstractVector, 
    P::AbstractMatrix, 
    c::AbstractVector, 
    T::AbstractMatrix, 
    Q::AbstractMatrix, 
    periods::Integer,
    samples::Integer,
    rng::AbstractRNG
)
    particles = similar(a, length(a), samples, periods)
    particles[:,:,1] = rand(rng, MvNormal(c + T * a, T * P * T' + Q), samples)
    for h = 2:periods, s = 1:samples
        particles[:,s,h] = rand(rng, MvNormal(c + T * particles[:,s,h-1], Q))
    end

    return particles
end

"""
    state_space(model) -> (y_star, Z_star, a1, P1)

State space form of the collapsed dynamic tensor autoregressive model `model`.
"""
function state_space(model::DynamicTensorAutoregression)
    dims = size(data(model))
    n = ndims(data(model)) - 1
    Ty = eltype(data(model))

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(model)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]

    # outer product of Kruskal factors
    U = [factors(model)[i+n] * factors(model)[i]' for i = 1:n]

    # scaling
    S = [Cinv[i] * U[i] for i = 1:n]

    # collapsing
    X = tucker(selectdim(data(model), n+1, 1:last(dims)-1), S)
    Z_star = [fill(norm(Xt), 1, 1) for Xt in eachslice(X, dims=n+1)]
    A_star = tucker(X, transpose.(Cinv))
    y_star = [vec(inv(Z_star[t]) * dot(vec(selectdim(A_star, n+1, t)), vec(selectdim(data(model), n+1, t+1)))) for t = 1:last(dims)-1]

    # initial conditions
    a1 = zeros(Ty, rank(model))
    P1 = Matrix{Ty}(I, rank(model), rank(model))

    return (y_star, Z_star, a1, P1)
end

"""
    filter(y, Z, c, T, Q, a1, P1) -> (a, P, v, F, K)

Collapsed Kalman filter for the dynamic tensor autoregressive model with system
matrices `Z`, `T`, and `Q`, state mean adjustment `c`, and initial conditions
`a1` and `P1`. Returns the filtered state `a`, covariance `P`, forecast error
`v`, forecast error variance `F`, and Kalman gain `K`.
"""
function filter(
    y::AbstractVector, 
    Z::AbstractVector,
    c::AbstractVector, 
    T::AbstractMatrix, 
    Q::AbstractMatrix, 
    a1::AbstractVector, 
    P1::AbstractMatrix
)
    a = similar(y, typeof(a1))
    P = similar(y, typeof(P1))
    v = similar(y)
    F = similar(y, typeof(P1))
    K = similar(y, typeof(P1))

    # initialize filter
    a[1] = a1
    P[1] = P1

    # filter
    for t ∈ eachindex(y)
        # forecast error
        v[t] = y[t] - Z[t] * a[t]
        F[t] = Z[t] * P[t] * Z[t]' + I

        # Kalman gain
        K[t] = T * P[t] * Z[t]' / F[t]

        # prediction
        if t < length(y)
            a[t+1] = T * a[t] + K[t] * v[t] + c
            P[t+1] = T * P[t] * (T - K[t] * Z[t])' + Q
        end
    end

    return (a, P, v, F, K)
end

"""
    smoother(y, Z, c, T, Q, a1, P1) -> (α̂, V, Γ)

Collapsed Kalman smoother for the dynamic tensor autoregressive model with
system matrices `Z`, `T`, and `Q`, state mean adjustment `c`, and initial
conditions `a1` and `P1`. Returns the smoothed state `α̂`, covariance `V`, and
autocovariance `Γ`.
"""
function smoother(
    y::AbstractVector, 
    Z::AbstractVector,
    c::AbstractVector, 
    T::AbstractMatrix, 
    Q::AbstractMatrix, 
    a1::AbstractVector, 
    P1::AbstractMatrix,
)
    # filter
    (a, P, v, F, K) = filter(y, Z, c, T, Q, a1, P1)

    α̂ = similar(a)
    V = similar(P)
    Γ = similar(P, length(y)-1)

    # initialize smoother
    r = zero(a1)
    N = zero(P1)
    L = similar(T)

    # smoother
    for t ∈ reverse(eachindex(y))
        L .= T - K[t] * Z[t]

        # backward recursion
        r .= Z[t]' / F[t] * v[t] + L' * r
        N .= Z[t]' / F[t] * Z[t] + L' * N * L

        # smoothing
        α̂[t] = a[t] + P[t] * r
        V[t] = P[t] - P[t] * N * P[t]
        t > 1 && (Γ[t-1] = I - P[t] * N)
        t < length(y) && (Γ[t] *= L * P[t])
    end

    return (α̂, V, Γ)
end

"""
    simulate(A, burn, rng) -> (A_sim, A_burn)

Simulate the dynamic loadings from the dynamic Kruskal coefficient tensor `A`
with a burn-in period of `burn` using the random number generator `rng`.
"""
function simulate(A::DynamicKruskal, burn::Integer, rng::AbstractRNG)
    A_burn = DynamicKruskal(
        similar(loadings(A), size(loadings(A), 1), burn),
        intercept(A), 
        dynamics(A), 
        cov(A), 
        factors(A), 
        rank(A)
    )
    dist = MvNormal(cov(A_burn))
    # burn-in
    for (t, λt) ∈ pairs(eachslice(loadings(A_burn), dims=2))
        if t == 1
            # initial condition
            λt .= rand(rng, dist)
        else
            λt .= intercept(A_burn) + dynamics(A_burn) * loadings(A_burn)[:,t-1] + rand(rng, dist)
        end
    end

    A_sim = similar(A)
    copyto!(A_sim, A)
    dist = MvNormal(cov(A_sim))
    # simulate
    for (t, λt) ∈ pairs(eachslice(loadings(A_sim), dims=2))
        λt_lag = t == 1 ? loadings(A_burn)[:,end] : loadings(A_sim)[:,t-1]
        λt .= intercept(A_sim) + dynamics(A_sim) * λt_lag + rand(rng, dist)
    end

    return (A_sim, A_burn)
end

"""
    simulate(ε, burn, rng) -> (ε_sim, ε_burn)

Simulate from the tensor error distribution `ε` with a burn-in period of `burn`
using the random number generator `rng`.
"""
simulate(ε::WhiteNoise, burn::Integer, rng::AbstractRNG) = error("simulating data from white noise error not supported.")
function simulate(ε::TensorNormal, burn::Integer, rng::AbstractRNG)
    dims = size(resid(ε))
    n = ndims(resid(ε)) - 1

    # Cholesky decompositions of Σᵢ
    C = [cholesky(Hermitian(Σi)).L for Σi ∈ cov(ε)]

    ε_burn = TensorNormal(
        similar(resid(ε), dims[1:n]..., burn), 
        cov(ε)
    )
    # burn-in
    for εt ∈ eachslice(resid(ε_burn), dims=n+1)
        # sample independent random normals and use tucker product with Cholesky 
        # decompositions
        εt .= tucker(randn(rng, dims[1:n]...), C)
    end

    ε_sim = similar(ε)
    copyto!(ε_sim, ε)
    # simulate
    for εt ∈ eachslice(resid(ε_sim), dims=n+1)
        # sample independent random normals and use tucker product with Cholesky 
        # decompositions
        εt .= tucker(randn(rng, dims[1:n]...), C)
    end

    return (ε_sim, ε_burn)
end