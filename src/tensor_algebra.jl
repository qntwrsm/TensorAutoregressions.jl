#=
tensor_algebra.jl

    Provides a collection of functions that implement tensor algebra operations.

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/03/02
=#

"""
    matricize(A, n) -> An

Matricize tensor `A` by unfolding along modes `n`.
"""
function matricize(A::AbstractArray, n)
    d = size(A)
    m = setdiff(1:ndims(A), n)
    perm = [n; m]

    return reshape(permutedims(A, perm), prod(d[n]), prod(d[m]))
end

"""
    tensorize(An, n, dims) -> A

Tensorize matrix `An` by folding along modes `n` with tensor dimensions `dims`.
"""
function tensorize(An::AbstractMatrix, n, dims)
    m = setdiff(1:length(dims), n)
    perm = invperm([n; m])

    return permutedims(reshape(An, dims[n]..., dims[m]...), perm)
end

"""
    tucker(G, A, n) -> T

Tucker operator along modes `n` of tensor `G` with matrices `A`.
"""
function tucker(G::AbstractArray, A::AbstractVector, n)
    d = collect(size(G))
    for (i, k) in enumerate(n)
        Gk = matricize(G, k)
        d[k] = size(A[i], 1)
        G = tensorize(A[i] * Gk, k, d)
    end

    return G
end
tucker(G::AbstractArray, A::AbstractVector) = tucker(G, A, 1:length(A))

"""
    I(n, R) -> Id

Identity tensor of `n` modes with mode size `R`.
"""
function (I::UniformScaling)(n::Integer, R::Integer)
    Id = zeros((R for _ in 1:n)...)
    for i in 1:R
        Id[repeat([i], n)...] = one(Float64)
    end

    return Id
end

"""
    cp(X, R; tolerance = 1e-4, max_iter = 1000) -> (Xhat, obj)

Rank `R` CP-decomposition of a tensor `X` obtained using alternating least squares using a
tolerance of `tolerance` and maximum number of iterations of `max_iter`.
"""
function cp(X::AbstractArray, R::Integer; tolerance::AbstractFloat = 1e-4,
            max_iter::Integer = 1000)
    # initialization
    U = [let Ui = randn(Ii, R)
             Ui ./ norm.(eachcol(Ui))'
         end
         for Ii in size(X)]
    Xhat = StaticKruskal(randn(R), U, R)

    # inner product
    UtU = transpose.(factors(Xhat)) .* factors(Xhat)

    # alternating least squares
    iter = 0
    obj = -Inf
    converged = false
    while !converged && iter < max_iter
        # update Kruskal factors and loadings
        for (k, Uk) in pairs(factors(Xhat))
            k_ = setdiff(1:ndims(X), k)
            V = ones(eltype(X), R, R)
            for j in k_
                V .*= UtU[j]
            end
            Xk = matricize(X, k)
            Z = factors(Xhat)[k_[end]]
            for j in Iterators.drop(Iterators.reverse(k_), 1)
                Z = hcat(kron.(eachcol(Z), eachcol(factors(Xhat)[j]))...)
            end
            Uk .= Xk * Z / V

            # normalize
            if iter == 0
                loadings(Xhat) .= norm.(eachcol(Uk))
            else
                loadings(Xhat) .= norm.(eachcol(Uk), Inf)
            end
            Uk ./= loadings(Xhat)'

            # update inner product
            UtU[k] = Uk' * Uk
        end

        # update objective function
        obj_prev = obj
        residual = X .- full(Xhat)
        obj = 1.0 - norm(residual) / norm(X)

        # convergence
        δ = abs(obj - obj_prev)
        converged = δ < tolerance

        # update iteration counter
        iter += 1
    end

    # fix signs of loadings
    idx = findall(λ -> λ < 0, loadings(Xhat))
    loadings(Xhat)[idx] .*= -one(eltype(X))
    factors(Xhat)[1][:, idx] .*= -one(eltype(X))

    # fix signs of factors
    signs = similar(X, ndims(X))
    for r in 1:R
        # find sign of largest factor elements
        for (k, Uk) in pairs(factors(Xhat))
            idx = argmax(abs.(Uk[:, r]))
            signs[k] = sign(Uk[idx, r])
        end

        # flip signs
        negatives = findall(s -> s == -1, signs)
        for i in 1:(length(negatives) - length(negatives) % 2)
            k = negatives[i]
            factors(Xhat)[k][:, r] .*= -one(eltype(X))
        end
    end

    return (Xhat, obj)
end
