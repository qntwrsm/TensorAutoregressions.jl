#=
utilities.jl

    Provides a collection of utility tools for working with tensor 
    autoregressive models, such as moving average representation and 
    orthogonalize responses. 

@author: Quint Wiersma <q.wiersma@vu.nl>

@date: 2023/07/02
=#

"""
    moving_average(A, n) -> Ψ

Moving average, ``MA(∞)``, representation of the tensor autoregressive model
with Kruskal coefficient tensor `A`, computed up to the `n`th term.
"""
function moving_average(A::StaticKruskal, n::Integer)
    dims = size(A)

    # matricize Kruskal tensor
    An = matricize(full(A), dims[1:end÷2])

    # moving average coefficients
    Ψ = zeros(dims, n+1)
    for h = 0:periods
        selectdim(Ψ, ndims(Ψ), h+1) .= tensorize(An^h, dims[1:end÷2], dims)
    end
    
    return Ψ
end