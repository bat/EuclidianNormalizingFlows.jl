# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


const ZERO = false


sum_ladjs(ladjs::Real) = ladjs
sum_ladjs(ladjs::AbstractVector{<:Real}) = sum(ladjs)
sum_ladjs(ladjs::AbstractMatrix{<:Real}) = vec(sum(ladjs, dims = 1))'


function similar_zeros(x::AbstractArray{<:Number}, sz::Dims)
    r = similar(x, sz)
    fill!(r, zero(eltype(x)))
end

function rrule(::typeof(similar_zeros), x::AbstractArray{<:Number}, sz::Dims)
    _similar_zeros_pullback(ΔΩ) = (NoTangent(), ZeroTangent(), NoTangent())
    return similar_zeros(x, sz), _similar_zeros_pullback
end


function similar_fill(value::T, x::AbstractArray{<:U}, sz::Dims) where {T<:Number,U<:Number}
    R = promote_type(T, U)
    v = convert(R, value)
    r = similar(x, R, sz)
    fill!(r, value)
end

function rrule(::typeof(similar_fill), value::T, x::AbstractArray{<:U}, sz::Dims) where {T<:Number,U<:Number}
    _similar_zeros_pullback(ΔΩ) = (NoTangent(), sum(unthunk(ΔΩ)), ZeroTangent(), NoTangent())
    return similar_zeros(x, sz), _similar_zeros_pullback
end
