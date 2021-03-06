# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


_dot(v::AbstractVector{<:Real}, x::AbstractVecOrMat{<:Real}) = (v' * x)


# x and y may alias:
function householder_trafo!(y::AbstractVecOrMat{<:Real}, v::AbstractVector{<:Real}, x::AbstractVecOrMat{<:Real})
    k = _dot(v, x) / _dot(v, v)
    y .= muladd.(-2 .* k, v, x)
end


function householder_trafo(v::AbstractVector{T}, x::AbstractVecOrMat{U}) where {T<:Real,U<:Real}
    R = promote_type(T, U)
    y = similar(x, R)
    householder_trafo!(y, v, x)
    return y
end


function householder_trafo_pullback_v(v, x, ΔΩ)
    inrm = inv(norm(v))

    # Readable version:
    #=
    # w == normalize(v):
    w = inrm .* v
    # pullback for I - 2 * (w*w')) * x:
    ∂w = -2 .* (ΔΩ .* dot(w, x) .+ x .* dot(w, ΔΩ))
    # pullback for normalize(v):
    ∂v = inrm .* (∂w .- w .* dot(∂w, w))
    =#

    inrm_2 = inrm * inrm
    w_x = inrm * _dot(v, x)
    w_ΔΩ = inrm * _dot(v, ΔΩ)
    ∂w_v = sum(-2 .* v .* (w_ΔΩ .* x .+ w_x .* ΔΩ), dims = 1)
    sum(inrm .* (-2 .* (w_x .* ΔΩ  .+ w_ΔΩ .* x) .- inrm_2 .* ∂w_v .* v), dims = 2)
end


#householder_trafo_pullback_x(v, x, ΔΩ) = householder_trafo(v, ΔΩ)

function householder_trafo_pbacc_x!(∂x, v, x, ΔΩ)
    k = -2 .* _dot(v, ΔΩ) ./ _dot(v, v)
    ∂x += muladd.(k, v, ΔΩ)
end

function householder_trafo_pullback_x(v, x, ΔΩ)
    R = promote_type(eltype(v), eltype(ΔΩ))
    ∂x = fill!(similar(x, R), 0)
    householder_trafo_pbacc_x!(∂x, v, x, ΔΩ)
end

function rrule(::typeof(householder_trafo), v::AbstractVector{T}, x::AbstractVecOrMat{U}) where {T<:Real,U<:Real}
    y = householder_trafo(v, x)
    function householder_trafo_pullback(ΔΩ)
        ∂v = @thunk(householder_trafo_pullback_v(v, x, unthunk(ΔΩ)))
        ∂x = InplaceableThunk(
            ∂x -> householder_trafo_pbacc_x!(∂x, v, x, unthunk(ΔΩ)),
            @thunk(householder_trafo_pullback_x(v, x, unthunk(ΔΩ)))
        )
        (NoTangent(), ∂v, ∂x)
    end
    return y, householder_trafo_pullback
end



function chained_householder_trafo!(y::AbstractVecOrMat{<:Real}, V::AbstractMatrix{<:Real}, x::AbstractVecOrMat{<:Real})
    y .= x    
    for i in axes(V, 2)
        v = view(V, :, i)
        householder_trafo!(y, v, y)
    end
    return y
end

function chained_householder_trafo(V::AbstractMatrix{T}, x::AbstractVecOrMat{U}) where {T<:Real,U<:Real}
    R = promote_type(T, U)
    y = similar(x, R)
    chained_householder_trafo!(y, V, x)
    return y
end


function chained_householder_trafo_pullback_V(V, x, y, ΔΩ)
    # @assert y == chained_householder_trafo(V, x)
    ∂V = similar(V)
    z = deepcopy(y)
    Δ = similar(x)
    Δ .= ΔΩ
    for i in reverse(axes(V, 2))
        v = view(V, :, i)
        ∂v = view(∂V, :, i)
        householder_trafo!(z, v, z)
        ∂v .= householder_trafo_pullback_v(v, z, Δ)
        householder_trafo!(Δ, v, Δ)
    end
    @assert z ≈ x
    return ∂V
end

function chained_householder_trafo_pullback_x(V, x, y, ΔΩ)
    # @assert y == chained_householder_trafo(V, x)
    ∂x = similar(x)
    ∂x .= ΔΩ
    for i in reverse(axes(V, 2))
        v = view(V, :, i)
        householder_trafo!(∂x, v, ∂x)
    end
    return ∂x
end

function rrule(::typeof(chained_householder_trafo), V::AbstractMatrix{T}, x::AbstractVecOrMat{U}) where {T<:Real,U<:Real}
    y = chained_householder_trafo(V, x)
    function householder_trafo_pullback(ΔΩ)
        ∂V = @thunk(chained_householder_trafo_pullback_V(V, x, y, unthunk(ΔΩ)))
        ∂x = @thunk(chained_householder_trafo_pullback_x(V, x, y, unthunk(ΔΩ)))
        (NoTangent(), ∂V, ∂x)
    end
    return y, householder_trafo_pullback
end


@with_kw struct HouseholderTrafo{T<:AbstractVecOrMat{<:Real}} <:Function
    V::T
end


# ToDo: Normalize in HouseholderTrafo as well?

_ht_normalize(V::AbstractVector{<:Real}) = normalize(V)
function _ht_normalize(V::AbstractMatrix{<:Real})
    V_norm = deepcopy(V)
    normalize!.(eachcol(V_norm))
    V_norm
end

function Base.convert(::Type{HouseholderTrafo}, nt::NamedTuple{(:V,),<:Tuple{AbstractVecOrMat{<:Real}}})
    HouseholderTrafo(_ht_normalize(nt.V))
end
Base.convert(::Type{NamedTuple}, x::HouseholderTrafo) = (V = x.V,)

Functors.functor(::Type{<:HouseholderTrafo}, x) = convert(NamedTuple, x), y -> convert(HouseholderTrafo, y)


Base.:(==)(a::HouseholderTrafo, b::HouseholderTrafo) = a.V == b.V
Base.isequal(a::HouseholderTrafo, b::HouseholderTrafo) = isequal(a.V, b.V)
Base.hash(x::HouseholderTrafo, h::UInt) = hash(x.V, hash(:HouseholderTrafo, hash(:EuclidianNormalizingFlows, h)))

inverse(f::HouseholderTrafo{<:AbstractVector}) = f
inverse(f::HouseholderTrafo{<:AbstractMatrix}) = HouseholderTrafo(reverse(f.V, dims = 2))

(f::HouseholderTrafo{<:AbstractVector})(x::AbstractVecOrMat{<:Real}) = householder_trafo(f.V, x)
(f::HouseholderTrafo{<:AbstractMatrix})(x::AbstractVecOrMat{<:Real}) = chained_householder_trafo(f.V, x)

with_logabsdet_jacobian(f::HouseholderTrafo, x::AbstractVector{T}) where {T<:Real} = (f(x), zero(T))
with_logabsdet_jacobian(f::HouseholderTrafo, x::AbstractMatrix{T}) where {T<:Real} = (f(x), similar_zeros(x, (size(x, 2),))')
