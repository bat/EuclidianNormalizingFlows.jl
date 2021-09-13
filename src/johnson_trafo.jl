struct JohnsonSU{T1<:Real,T2<:Real,T3<:Real,T4<:Real} <: Distributions.Distribution{Univariate,Continuous}
    gamma::T1 #positive: left tail, negativ: right tail, 0: no tail
    delta::T2 #height and width
    xi::T3 #shift along the x-axis
    lambda::T4 #height and width
end


function JohnsonSU(; gamma::Real = 10, delta::Real = 3.5, xi::Real = 10, lambda::Real = 1)
    gamma, delta, xi, lambda = promote(gamma, delta, xi, lambda)
    JohnsonSU(gamma, delta, xi, lambda)
end


Base.minimum(d::JohnsonSU) = -Inf
Base.maximum(d::JohnsonSU) = +Inf

StatsBase.params(d::JohnsonSU) = (d.gamma, d.delta, d.xi, d.lambda)
@inline Distributions.partype(d::JohnsonSU{T}) where {T<:Real} = T

Distributions.location(d::JohnsonSU) = mean(d)
Distributions.scale(d::JohnsonSU) = var(d)

Statistics.mean(d::JohnsonSU) = d.xi - d.lambda * exp(d.delta^(-2)/2)*sinh(d.gamma/d.delta)
Statistics.median(d::JohnsonSU) = d.xi + d.lambda * sinh(-d.gamma/d.delta)
Statistics.var(d::JohnsonSU) = (d.lambda^2)/2 * (exp(d.delta^(-2)) - 1) * (exp(d.delta^(-2))*cosh(2*d.gamma/d.delta) + 1)


function johnsontrafo(x::T1, gamma::T2, delta::T3, xi::T4, lambda::T5) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real, T5<:Real}
    R = float(promote_type(T1, T2, T3, T4, T5))
    convert(R, (gamma + delta*asinh((x - xi)/lambda)))
end

function johnsontrafo_inv(x::T1, gamma::T2, delta::T3, xi::T4, lambda::T5) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real, T5<:Real}
    R = float(promote_type(T1, T2, T3, T4, T5))
    convert(R, lambda*sinh((x - gamma)/delta) + xi)
end

function deriv_johnsontrafo(x::T1, gamma::T2, delta::T3, xi::T4, lambda::T5) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real, T5<:Real}
    R = float(promote_type(T1, T2, T3, T4, T5))
    convert(R, (delta/lambda) * (1/(sqrt(1 + ((x - xi)/lambda)^2))))
end

function deriv_johnsontrafo_inv(x::T1, gamma::T2, delta::T3, xi::T4, lambda::T5) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real, T5<:Real}
    R = float(promote_type(T1, T2, T3, T4, T5))
    convert(R, lambda*cosh((x - gamma)/delta)/delta)
end

function johnsontrafo_ladj(x::T1, gamma::T2, delta::T3, xi::T4, lambda::T5) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real, T5<:Real}
    R = float(promote_type(T1, T2, T3, T4, T5))
    convert(R, log(abs(deriv_johnsontrafo(x,gamma,delta,xi,lambda))))
end

function johnsontrafo_inv_ladj(x::T1, gamma::T2, delta::T3, xi::T4, lambda::T5) where {T1<:Real,T2<:Real,T3<:Real,T4<:Real, T5<:Real}
    R = float(promote_type(T1, T2, T3, T4, T5))
    convert(R, log(abs(deriv_johnsontrafo_inv(x,gamma,delta,xi,lambda))))
end

@with_kw struct JohnsonTrafo{T<:Union{Real,AbstractArray{<:Real}}} <: Function
    gamma::T = 10.0
    delta::T = 3.5
    xi::T = 10.0
    lambda::T = 1.0
end

@functor JohnsonTrafo

#(f::JohnsonTrafo)(x::Union{Real,AbstractArray{<:Real}}) = fwddiff(johnsontrafo).(x, f.gamma, f.delta, f.xi, f.lambda)
(f::JohnsonTrafo)(x::Union{Real,AbstractArray{<:Real}}) = johnsontrafo.(x, f.gamma, f.delta, f.xi, f.lambda)

function (f::JohnsonTrafo)(x::Union{Real,AbstractArray{<:Real}}, ::WithLADJ)
    y = f(x)
    ladjs = johnsontrafo_ladj.(x, f.gamma, f.delta, f.xi, f.lambda)
    (y, sum(ladjs))
end

Base.inv(f::JohnsonTrafo) = JohnsonTrafoInv(f.gamma, f.delta, f.xi, f.lambda)

@with_kw struct JohnsonTrafoInv{T<:Union{Real,AbstractArray{<:Real}}} <: Function
    gamma::T = 10.0
    delta::T = 3.5
    xi::T = 10.0
    lambda::T = 1.0
end

@functor JohnsonTrafoInv

#(f::JohnsonTrafoInv)(x::Union{Real,AbstractArray{<:Real}}) = fwddiff(johnsontrafo_inv).(x, f.gamma, f.delta, f.xi, f.lambda)
(f::JohnsonTrafoInv)(x::Union{Real,AbstractArray{<:Real}}) = johnsontrafo_inv.(x, f.gamma, f.delta, f.xi, f.lambda)

function (f::JohnsonTrafoInv)(x::Union{Real,AbstractArray{<:Real}}, ::WithLADJ)
    y = f(x)
    ladjs = johnsontrafo_inv_ladj.(x, f.gamma, f.delta, f.xi, f.lambda)
    (y, sum(ladjs))
end

Base.inv(f::JohnsonTrafoInv) = JohnsonTrafo(f.gamma, f.delta, f.xi, f.lambda)

#johnsontrafo(d::JohnsonSU, x::Real) = (d.gamma + d.delta*asinh((x - d.xi)/d.lambda))
#johnsontrafo_inv(d::JohnsonSU, x::Real) = d.lambda*sinh((x - d.gamma)/d.delta) + d.xi
#deriv_johnsontrafo(d::JohnsonSU, x::Real) = (d.delta/d.lambda) * (1/(sqrt(1 + ((x - d.xi)/d.lambda)^2)))
#deriv_johnsontrafo_inv(d::JohnsonSU, x::Real) = d.lambda*cosh((x - d.gamma)/d.delta)/d.delta
#johnsontrafo_ladj(d::JohnsonSU, x::Real) = log(abs(deriv_johnsontrafo(d::JohnsonSU, x::Real)))
#johnsontrafo_inv_ladj(d::JohnsonSU, x::Real) = log(abs(deriv_johnsontrafo_inv(d::JohnsonSU, x::Real)))


#Distributions.pdf(d::JohnsonSU, x::Real) = deriv_johnsontrafo(d,x) * 1/sqrt(2*pi) * exp(-0.5 * johnsontrafo(d,x)^2)
Distributions.pdf(d::JohnsonSU, x::Real) = deriv_johnsontrafo(x, d.gamma, d.delta, d.xi, d.lambda) * pdf(Normal(), johnsontrafo(x, d.gamma, d.delta, d.xi, d.lambda))
Distributions.cdf(d::JohnsonSU, x::Real) = cdf(Normal(), johnsontrafo(x, d.gamma, d.delta, d.xi, d.lambda))
#Distributions.logpdf(d::JohnsonSU, x::Real) = log(deriv_johnsontrafo(d,x) * 1/sqrt(2*pi) * exp(-0.5 * johnsontrafo(d,x)^2))
Distributions.logpdf(d::JohnsonSU, x::Real) = log(deriv_johnsontrafo(x, d.gamma, d.delta, d.xi, d.lambda) * pdf(Normal(), johnsontrafo(x, d.gamma, d.delta, d.xi, d.lambda)))
Distributions.logcdf(d::JohnsonSU, x::Real) = logcdf(Normal(), johnsontrafo(x, d.gamma, d.delta, d.xi, d.lambda))
Distributions.ccdf(d::JohnsonSU, x::Real) = 1 - Distributions.cdf(d, x)
Distributions.logccdf(d::JohnsonSU, x::Real) = log(1 - Distributions.cdf(d, x))


Statistics.quantile(d::JohnsonSU, x::Real) = johnsontrafo_inv(quantile(Normal(),x), d.gamma, d.delta, d.xi, d.lambda)

