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
    confidence_bounds(model, periods, α, orth, samples=1000, burn=100, rng=Xoshiro()) -> (lower, upper)

Compute Monte Carlo `α`% confidence bounds for impulse response functions of the
tensor autoregressive model given by `model`. The confidence boudns are
estimated using a Monte Carlo simulation with `samples` and a burn-in period
`burn`.
"""
function confidence_bounds(
    model::TensorAutoregression, 
    periods::Integer,
    α::Real, 
    orth::Bool, 
    samples::Integer=1000, 
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
        if coef(model) isa StaticKruskal
            Ψ[s] = moving_average(coef(sim), periods)
        else
            Ψ[s] = moving_average(coef(sim), periods, data(sim), dist(sim))
        end

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
    moving_average(A, n[, y, ε]) -> Ψ

Moving average, ``MA(∞)``, representation of the tensor autoregressive model
with Kruskal coefficient tensor `A`, computed up to the `n`th term.
"""
function moving_average(A::StaticKruskal, n::Integer)
    dims = size(A)

    # matricize Kruskal tensor
    An = matricize(full(A), 1:length(dims)÷2)

    # moving average coefficients
    Ψ = zeros(dims..., n+1)
    for (h, ψh) ∈ pairs(eachslice(Ψ, dims=ndims(Ψ)))
        ψh .= tensorize(An^(h-1), 1:ndims(ψh)÷2, dims)
    end

    return Ψ
end

function moving_average(A::DynamicKruskal, n::Integer, y::AbstractArray, ε::TensorNormal)
    dims = size(A)

    # tensorize identity matrix
    Id = tensorize(I(prod(size(y)[1:end-1])), 1:(length(dims)-1)÷2, dims[1:end-1])

    # matricize Kruskal tensor
    An = matricize(full(A), 1:(length(dims)-1)÷2)

    # moving average coefficients
    Ψ = zeros(dims[1:end-1]..., n+1, last(dims))
    for (t, ψt) ∈ pairs(eachslice(Ψ, dims=ndims(Ψ)))
        # sample particles
        particles = get_particles(selectdim(y, ndims(y), 1:t+1), A, ε, n)
        for (h, ψh) ∈ pairs(eachslice(ψt, dims=ndims(ψt)))
            if h == 1
                ψh .= Id 
            else
                λ = mean(prod(particles[1,:,1:h-1], dims=2))
                ψh .= λ * tensorize(An^(h-1), 1:ndims(ψh)÷2, size(ψh))
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
    # Cholesky decomposition of Σ
    C = cholesky(Hermitian(Σ))

    # orthogonalize responses
    Ψ_orth = similar(Ψ)
    for (h, ψ) ∈ pairs(eachslice(Ψ, dims=ndims(Ψ)))
        selectdim(Ψ_orth, ndims(Ψ_orth), h) .= tensorize(matricize(ψ, 1:ndims(ψ)÷2) * C.L, 1:ndims(ψ)÷2, size(ψ))
    end

    return Ψ_orth
end

function orthogonalize(Ψ::AbstractArray, Σ::AbstractVector)
    # Cholesky decompositions of Σᵢ
    C = [cholesky(Hermitian(Σi)).L for Σi ∈ Σ]

    # orthogonalize responses
    Ψ_orth = tucker(Ψ, C, 1:length(C))

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
    (a, P, v, _, K) = filter(y_star, Z_star, dynamics(A), cov(A), a1, P1)
    # predict
    â = dynamics(A) * a[end] + K[end] * v[end]
    P̂ = dynamics(A) * P[end] * (dynamics(A) - K[end] * Z_star[end])' + cov(A)

    # sample particles
    particles = particle_sampler(â, P̂, dynamics(A), cov(A), periods, 1000, Xoshiro())

    return particles
end

"""
    particle_sampler(a, P, T, Q, periods, samples, rng) -> particles

Forward particle sampler of the filtered state `a` with corresponding variance
`P` and state equation system matrices `T` and `Q` with the number of forward
periods given by `periods`, using random number generator `rng`.
"""
function particle_sampler(
    a::AbstractVector, 
    P::AbstractMatrix, 
    T::AbstractMatrix, 
    Q::AbstractMatrix, 
    periods::Integer,
    samples::Integer,
    rng::AbstractRNG
)
    particles = similar(a, length(a), samples, periods)
    particles[:,:,1] = rand(rng, MvNormal(a, P), samples)
    for h = 2:periods, s = 1:samples
        particles[:,s,h] = rand(rng, MvNormal(T * particles[:,s,h-1], Q))
    end

    return particles
end

"""
    state_space(y, A, ε) -> (y_star, Z_star, a1, P1)

State space form of the collapsed tensor autoregressive model with dynamic
Kruskal coefficient tensor `A` and tensor error distribution `ε`.
"""
function state_space(y::AbstractArray, A::DynamicKruskal, ε::TensorNormal)
    dims = size(y)
    n = ndims(y) - 1

    # Cholesky decompositions of Σᵢ
    C = cholesky.(Hermitian.(cov(ε)))
    # inverse of Cholesky decompositions
    Cinv = [inv(C[i].L) for i = 1:n]

    # outer product of Kruskal factors
    U = [factors(A)[i] * factors(A)[i+n]' for i = 1:n]

    # scaling
    S = [Cinv[i] * U[i] for i = 1:n]

    # collapsing
    X = tucker(selectdim(y, n+1, 1:last(dims)-1), S, 1:n)
    Z_star = [fill(norm(Xt), 1, 1) for Xt in eachslice(X, dims=n+1)]
    A_star = tucker(X, transpose.(Cinv), 1:n)
    y_star = [vec(inv(Z_star[t]) * dot(vec(selectdim(A_star, n+1, t)), vec(selectdim(y, n+1, t+1)))) for t = 1:last(dims)-1]

    # initial conditions
    a1 = zeros(eltype(y), rank(A))
    P1 = Matrix{eltype(y)}(I, rank(A), rank(A))

    return (y_star, Z_star, a1, P1)
end

"""
    filter(y, Z, T, Q, a1, P1) -> (a, P, v, F, K)

Collapsed Kalman filter for the dynamic tensor autoregressive model with system
matrices `Z`, `T`, and `Q` and initial conditions `a1` and `P1`. Returns the
filtered state `a`, covariance `P`, forecast error `v`, forecast error variance
`F`, and Kalman gain `K`.
"""
function filter(
    y::AbstractVector, 
    Z::AbstractVector, 
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
            a[t+1] = T * a[t] + K[t] * v[t]
            P[t+1] = T * P[t] * (T - K[t] * Z[t])' + Q
        end
    end

    return (a, P, v, F, K)
end

"""
    smoother(y, Z, T, Q, a1, P1) -> (α̂, V, Γ)

Collapsed Kalman smoother for the dynamic tensor autoregressive model with
system matrices `Z`, `T`, and `Q` and initial conditions `a1` and `P1`. Returns
the smoothed state `α̂`, covariance `V`, and autocovariance `Γ`.
"""
function smoother(
    y::AbstractVector, 
    Z::AbstractVector, 
    T::AbstractMatrix, 
    Q::AbstractMatrix, 
    a1::AbstractVector, 
    P1::AbstractMatrix,
)
    # filter
    (a, P, v, F, K) = filter(y, Z, T, Q, a1, P1)

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
            λt .= dynamics(A_burn) * loadings(A_burn)[:,t-1] + rand(rng, dist)
        end
    end

    A_sim = similar(A)
    copyto!(A_sim, A)
    dist = MvNormal(cov(A_sim))
    # simulate
    for (t, λt) ∈ pairs(eachslice(loadings(A_sim), dims=2))
        λt_lag = t == 1 ? loadings(A_burn)[:,end] : loadings(A_sim)[:,t-1]
        λt .= dynamics(A_sim) * λt_lag + rand(rng, dist)
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
    for εt ∈ eachslice(resid(ε_sim), dims=n+1)
        # sample independent random normals and use tucker product with Cholesky 
        # decompositions
        εt .= tucker(randn(rng, dims[1:n]...), C, 1:n)
    end

    ε_sim = similar(ε)
    copyto!(ε_sim, ε)
    # simulate
    for εt ∈ eachslice(resid(ε_sim), dims=n+1)
        # sample independent random normals and use tucker product with Cholesky 
        # decompositions
        εt .= tucker(randn(rng, dims[1:n]...), C, 1:n)
    end

    return (ε_sim, ε_burn)
end