# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


# x and y may alias:
function householder_trafo!(y::AbstractVector{<:Real}, v::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    k = - 2 * dot(v, x) / dot(v, v)
    y .= muladd.(k, v, x)
end


function householder_trafo(v::AbstractVector{T}, x::AbstractVector{U}) where {T<:Real,U<:Real}
    R = promote_type(T, U)
    y = similar(x, R)
    householder_trafo!(y, v, x)
    return y
end


# ∂v and v, x, ΔΩ must not alias
function householder_trafo_pbacc_v!(∂v, v, x, ΔΩ)
    inrm = inv(norm(v))

    # Readable version:
    #=
    # w == normalize(v):
    w = inrm .* v
    # pullback for I - 2 * (w*w')) * x:
    ∂w = -2 .* (ΔΩ .* dot(w, x) .+ x .* dot(w, ΔΩ))
    # pullback for normalize(v):
    ∂v .+= inrm .* (∂w .- w .* dot(∂w, w))
    =#

    inrm_2 = inrm * inrm
    w_x = inrm * dot(v, x)
    w_ΔΩ = inrm * dot(v, ΔΩ)
    _∂w_v_i(v_i, x_i, ΔΩ_i) = -2 * v_i * (w_ΔΩ * x_i + w_x * ΔΩ_i)
    ∂w_v = sum(Base.Broadcast.broadcasted(_∂w_v_i, v, x, ΔΩ))
    ∂v .+= inrm .* (-2 .* (w_x .* ΔΩ  .+ w_ΔΩ .* x) .- inrm_2 .* ∂w_v .* v)

    return ∂v
end

function householder_trafo_pullback_v(v, x, ΔΩ)
    R = promote_type(eltype(v), eltype(x), eltype(ΔΩ))
    ∂v = fill!(similar(v, R), 0)
    householder_trafo_pbacc_v!(∂v, v, x, ΔΩ)
end


#householder_trafo_pullback_x(v, x, ΔΩ) = householder_trafo(v, ΔΩ)

function householder_trafo_pbacc_x!(∂x, v, x, ΔΩ)
    k = - 2 * dot(v, ΔΩ) / dot(v, v)
    ∂x += muladd.(k, v, ΔΩ)
end

function householder_trafo_pullback_x(v, x, ΔΩ)
    R = promote_type(eltype(v), eltype(ΔΩ))
    ∂x = fill!(similar(x, R), 0)
    householder_trafo_pbacc_x!(∂x, v, x, ΔΩ)
end

function ChainRulesCore.rrule(::typeof(householder_trafo), v::AbstractVector{T}, x::AbstractVector{U}) where {T<:Real,U<:Real}
    y = householder_trafo(v, x)
    function householder_trafo_pullback(ΔΩ)
        ∂v = InplaceableThunk(
            @thunk(householder_trafo_pullback_v(v, x, unthunk(ΔΩ))),
            ∂v -> householder_trafo_pbacc_v!(∂v, v, x, unthunk(ΔΩ))
        )
        ∂x = InplaceableThunk(
            @thunk(householder_trafo_pullback_x(v, x, unthunk(ΔΩ))),
            ∂x -> householder_trafo_pbacc_x!(∂x, v, x, unthunk(ΔΩ))
        )
        (NoTangent(), ∂v, ∂x)
    end
    return y, householder_trafo_pullback
end



function chained_householder_trafo!(y::AbstractVector{<:Real}, V::AbstractMatrix{<:Real}, x::AbstractVector{<:Real})
    y .= x    
    for i in axes(V, 2)
        v = view(V, :, i)
        householder_trafo!(y, v, y)
    end
    return y
end

function chained_householder_trafo(V::AbstractMatrix{T}, x::AbstractVector{U}) where {T<:Real,U<:Real}
    R = promote_type(T, U)
    y = similar(x, R)
    chained_householder_trafo!(y, V, x)
    return y
end


function chained_householder_trafo_pullback_V(V, x, y, ΔΩ)
    # @assert y == chained_householder_trafo(V, x)
    ∂V = similar(V)
    z = deepcopy(y)
    Δ = deepcopy(ΔΩ)
    for i in reverse(axes(V, 2))
        v = view(V, :, i)
        ∂v = view(∂V, :, i)
        householder_trafo!(z, v, z)
        ∂v .= 0
        householder_trafo_pbacc_v!(∂v, v, z, Δ)
        householder_trafo!(Δ, v, Δ)
    end
    @assert z ≈ x
    return ∂V
end

function chained_householder_trafo_pullback_x(V, x, y, ΔΩ)
    # @assert y == chained_householder_trafo(V, x)
    ∂x = deepcopy(ΔΩ)
    for i in reverse(axes(V, 2))
        v = view(V, :, i)
        householder_trafo!(∂x, v, ∂x)
    end
    return ∂x
end

function ChainRulesCore.rrule(::typeof(chained_householder_trafo), V::AbstractMatrix{T}, x::AbstractVector{U}) where {T<:Real,U<:Real}
    y = chained_householder_trafo(V, x)
    function householder_trafo_pullback(ΔΩ)
        ∂V = @thunk(chained_householder_trafo_pullback_V(V, x, y, unthunk(ΔΩ)))
        ∂x = @thunk(chained_householder_trafo_pullback_x(V, x, y, unthunk(ΔΩ)))
        (NoTangent(), ∂V, ∂x)
    end
    return y, householder_trafo_pullback
end
