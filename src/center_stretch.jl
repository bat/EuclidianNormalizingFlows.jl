# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


function center_stretch(x::T1, a::T2, b::T3, c::T4) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    R = float(promote_type(T1, T2, T3, T4))
    exp_abs_bx = exp(abs(b * x))
    convert(R, sign(x) * log((sqrt((1 - exp_abs_bx)^2 * exp(2 * b*a) + 4* exp_abs_bx) - (1 - exp_abs_bx) * exp(b*a))/2)/b + c)
end


function center_contract(x::T1, a::T2, b::T3, c::T4) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    R = float(promote_type(T1, T2, T3, T4))
    x_unshifted = x - c
    convert(R, (log(1 + exp(b * (x_unshifted-a))) - log(1 + exp(-b * (x_unshifted+a)))) / b)
end

function center_contract_ladj(x::T1, a::T2, b::T3, c::T4) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real}
    R = float(promote_type(T1, T2, T3, T4))
    x_unshifted = x - c
    exp_b_x_minus_a = exp(b * (x_unshifted - a))
    exp_neg_b_x_plus_x = exp(-b * (x + a))
    dy_dx = 1/(1 + exp(-b * (x_unshifted - a))) + 1/(1 + exp(b * (x + a)))
    log(abs(dy_dx))
end


@with_kw struct CenterStretch{T<:Union{Real,AbstractArray{<:Real}}} <: Function
    a::T = 0.0
    b::T = 1.0
    c::T = 0.0
end

@functor CenterStretch

(f::CenterStretch)(x::Union{Real,AbstractArray{<:Real}}) = fwddiff(center_stretch).(x, f.a, f.b, f.c)

function (f::CenterStretch)(x::Union{Real,AbstractArray{<:Real}}, ::WithLADJ)
    y = f(x)
    neg_ladjs = fwddiff(center_contract_ladj).(y, f.a, f.b, f.c)
    (y, - sum(neg_ladjs))
end

Base.inv(f::CenterStretch) = CenterContract(f.a, f.b, f.c)



@with_kw struct CenterContract{T<:Union{Real,AbstractArray{<:Real}}} <:Function
    a::T = 0.0
    b::T = 1.0
    c::T = 0.0
end

@functor CenterContract

(f::CenterContract)(x::Union{Real,AbstractArray{<:Real}}) = fwddiff(center_contract).(x, f.a, f.b, f.c)

function (f::CenterContract)(x::Union{Real,AbstractArray{<:Real}}, ::WithLADJ)
    y = f(x)
    ladjs = center_contract_ladj.(x, f.a, f.b, f.c)
    (y, sum(ladjs))
end

Base.inv(f::CenterContract) = CenterStretch(f.a, f.b, f.c)
