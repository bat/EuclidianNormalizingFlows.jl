# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


const ZERO = false


sum_ladjs(ladjs::Real) = ladjs
sum_ladjs(ladjs::AbstractVector{<:Real}) = sum(ladjs)
sum_ladjs(ladjs::AbstractMatrix{<:Real}) = vec(sum(ladjs, dims = 1))


function similar_zeros(x::AbstractMatrix{<:Number}, sz::Dims)
    r = similar(x, sz)
    fill!(r, zero(eltype(x)))
end

function rrule(::typeof(similar_zeros), x::AbstractMatrix{<:Number}, sz::Dims)
    _similar_zeros_pullback(ΔΩ) = (NoTangent(), ZeroTangent(), NoTangent())
    return similar_zeros(x, sz), _similar_zeros_pullback
end
