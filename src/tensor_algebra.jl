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
    dims = size(A)
    m = setdiff(1:ndims(A), n)
    perm = [n; m]

    return reshape(permutedims(A, perm), prod(dims[n]), prod(dims[m])) 
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
    tucker(G, A) -> T

Tucker operator of tensor `G` with matrices `A`.
"""
function tucker(G::AbstractArray, A::AbstractVector)
    dims = collect(size(G))
    for (i, Ai) ∈ enumerate(A)
        Gi = matricize(G, i)
        dims[i] = size(Ai, 1)
        G = tensorize(Ai * Gi, i, dims)
    end

    return G
end

"""
    I(n, R) -> Id

Identity tensor of `n` modes with mode size `R`.
"""
function (I::UniformScaling)(n::Integer, R::Integer)
    Id = zeros((n for _ = 1:R)...)
    for i = 1:R
        Id[repeat([i], n)...] = one(Float64)
    end

    return Id
end