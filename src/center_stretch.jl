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
