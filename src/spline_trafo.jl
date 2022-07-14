# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).
# The algorithm implemented here is described in https://arxiv.org/abs/1906.04032 

struct RationalQuadSpline <: Function
    widths::AbstractMatrix{<:Real}
    heights::AbstractMatrix{<:Real}
    derivatives::AbstractMatrix{<:Real}
end

export RationalQuadSpline
@functor RationalQuadSpline

struct RationalQuadSplineInv <: Function
    widths::AbstractMatrix{<:Real}
    heights::AbstractMatrix{<:Real}
    derivatives::AbstractMatrix{<:Real}
end

@functor RationalQuadSplineInv
export RationalQuadSplineInv


Base.:(==)(a::RationalQuadSpline, b::RationalQuadSpline) = a.widths == b.widths && a.heights == b.heights && a.derivatives == b.derivatives

Base.isequal(a::RationalQuadSpline, b::RationalQuadSpline) = isequal(a.widths, b.widths) && isequal(a.heights, b.heights) && isequal(a.derivatives, b.derivatives)

Base.hash(x::RationalQuadSpline, h::UInt) = hash(x.widths, hash(x.heights, hash(x.derivatives, hash(:RationalQuadSpline, hash(:EuclidianNormalizingFlows, h)))))

(f::RationalQuadSpline)(x::AbstractMatrix{<:Real}) = spline_forward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RationalQuadSpline,
    x::AbstractMatrix{<:Real}
)
    return spline_forward(f, x)
end

function InverseFunctions.inverse(f::RationalQuadSpline)
    return RationalQuadSplineInv(f.widths, f.heights, f.derivatives)
end

Base.:(==)(a::RationalQuadSplineInv, b::RationalQuadSplineInv) = a.widths == b.widths && a.heights == b.heights && a.derivatives == b.derivatives

Base.isequal(a::RationalQuadSplineInv, b::RationalQuadSplineInv) = isequal(a.widths, b.widths) && isequal(a.heights, b.heights) && isequal(a.derivatives, b.derivatives)

Base.hash(x::RationalQuadSplineInv, h::UInt) = hash(x.widths, hash(x.heights, hash(x.derivatives, hash(:RationalQuadSplineInv, hash(:EuclidianNormalizingFlows, h)))))

(f::RationalQuadSplineInv)(x::AbstractMatrix{<:Real}) = spline_backward(f, x)[1]

function ChangesOfVariables.with_logabsdet_jacobian(
    f::RationalQuadSplineInv,
    x::AbstractMatrix{<:Real}
)
    return spline_backward(f, x)
end

function InverseFunctions.inverse(f::RationalQuadSplineInv)
    return RationalQuadSpline(f.widths, f.heights, f.derivatives)
end

# Transformation forward: 

function spline_forward(trafo::RationalQuadSpline, x::AbstractMatrix{<:Real})
    
    nsmpls = size(x, 2)

    @assert size(trafo.widths, 1) == size(trafo.heights, 1) == size(trafo.derivatives, 1) == size(x, 1) >= 1
    @assert size(trafo.widths, 2) == size(trafo.heights, 2) == (size(trafo.derivatives, 2) + 1) >= 2

    w = cumsum_with_jac_mat(softmax_with_jac_mat(trafo.widths))
    h = cumsum_with_jac_mat(softmax_with_jac_mat(trafo.heights))
    d = softplus_with_jac_mat(trafo.derivatives)

    return spline_forward(x, w, h, d, w, h, d)
end

function spline_forward(
    x::AbstractArray{M0},
    w::AbstractArray{M1},
    h::AbstractArray{M2},
    d::AbstractArray{M3},
    w_logJac::AbstractArray{M4},
    h_logJac::AbstractArray{M5},
    d_logJac::AbstractArray{M6};
    B=5.
) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real, M6<:Real}

    T = promote_type(M0, M1, M2, M3, M4, M5, M6)

    ndims = size(x, 1)
    nsmpls = size(x, 2)
    
    B = fill(B, size(w, 1))
    one = ones(size(d, 1))

    # add boundary conditions:
    w = convert(ElasticArray{eltype(w)}, w)
    h = convert(ElasticArray{eltype(h)}, h)
    d = convert(ElasticArray{eltype(d)}, d)

    prepend!(w, -B)
    prepend!(h, -B)
    prepend!(d, one)
    append!(d, one)

    y = zeros(T, ndims, nsmpls)
    LogJac = zeros(T, 1, nsmpls)

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : 4
    kernel! = spline_forward_kernel!(device, n)

    ev = kernel!(x, y, LogJac, w, h, d, ndrange=size(x))

    wait(ev)

    return y, LogJac
end


function spline_forward_pullback(
        x::AbstractArray{M0},
        w::AbstractArray{M1},
        h::AbstractArray{M2},
        d::AbstractArray{M3},
        w_logJac::AbstractArray{M4},
        h_logJac::AbstractArray{M5},
        d_logJac::AbstractArray{M6},
        tangent::ChainRulesCore.Tangent;
        B=5.
    ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real, M6<:Real}

    T = promote_type(M0, M1, M2, M3, M4, M5, M6)

    ndims = size(x, 1)
    nsmpls = size(x, 2)
    nparams = size(w, 2)

    B = fill(B, size(w, 1))
    one = ones(size(d, 1))

    # add boundary conditions:
    w = convert(ElasticArray{eltype(w)}, w)
    h = convert(ElasticArray{eltype(h)}, h)
    d = convert(ElasticArray{eltype(d)}, d)

    prepend!(w, -B)
    prepend!(h, -B)
    prepend!(d, one)
    append!(d, one)

    y = zeros(T, ndims, nsmpls)
    LogJac = zeros(T, 1, nsmpls)

    ∂y∂w = zeros(T, ndims, nparams)
    ∂y∂h = zeros(T, ndims, nparams)
    ∂y∂d = zeros(T, ndims, nparams-1)

    ∂LogJac∂w = zeros(T, ndims, nparams)
    ∂LogJac∂h = zeros(T, ndims, nparams)
    ∂LogJac∂d = zeros(T, ndims, nparams-1)

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : 4
    kernel! = spline_forward_pullback_kernel!(device, n)

    ev = kernel!(
        x, y, LogJac,
        w, h, d,
        ∂y∂w, ∂y∂h, ∂y∂d,
        ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d, 
        tangent,
        ndrange=size(x)
        )

    wait(ev)

    return NoTangent(), @thunk(tangent[1] .* exp.(LogJac)), ∂y∂w, ∂y∂h, ∂y∂d, ∂LogJac∂w, ∂LogJac∂h, ∂LogJac∂d
end

@kernel function spline_forward_kernel!(
    x::AbstractArray,
    y::AbstractArray,
    LogJac::AbstractArray,
    w::AbstractArray,
    h::AbstractArray,
    d::AbstractArray
)
    i, j = @index(Global, NTuple)

    K = size(w, 2)

    # Find the bin index
    k1 = searchsortedfirst_impl(w[i,:], x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is outside of range
    isoutside = (k1 >= K) || (k1 == 0)
    k = Base.ifelse(isoutside, k2, k1)

    x_tmp = Base.ifelse(isoutside, w[i,k], x[i,j]) # Simplifies unnecessary calculations
    (yᵢⱼ, LogJacᵢⱼ) = eval_forward_spline_params(w[i,k], w[i,k+1], h[i,k], h[i,k+1], d[i,k], d[i,k+1], x_tmp)

    @atomic y[i,j] = Base.ifelse(isoutside, x[i,j], yᵢⱼ) 
    @atomic LogJac[1, j] += Base.ifelse(isoutside, zero(typeof(LogJacᵢⱼ)), LogJacᵢⱼ)

end


@kernel function spline_forward_pullback_kernel!(
        x::AbstractArray,
        y::AbstractArray,
        LogJac::AbstractArray,
        w::AbstractArray,
        h::AbstractArray,
        d::AbstractArray,
        ∂y∂w::AbstractArray,
        ∂y∂h::AbstractArray,
        ∂y∂d::AbstractArray,
        ∂LogJac∂w::AbstractArray,
        ∂LogJac∂h::AbstractArray,
        ∂LogJac∂d::AbstractArray,
        tangent::ChainRulesCore.Tangent
    )

    i, j = @index(Global, NTuple)

    K = size(w, 2)

    # Find the bin index
    k1 = searchsortedfirst_impl(w[i,:], x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is outside of range
    isoutside = (k1 >= K) || (k1 == 0)
    k = Base.ifelse(isoutside, k2, k1)

    x_tmp = Base.ifelse(isoutside, w[i,k], x[i,j]) # Simplifies unnecessary calculations
    (yᵢⱼ, LogJacᵢⱼ, ∂y∂wₖ, ∂y∂hₖ, ∂y∂dₖ, ∂LogJac∂wₖ, ∂LogJac∂hₖ, ∂LogJac∂dₖ) = eval_forward_spline_params_with_grad(w[i,k], w[i,k+1], h[i,k], h[i,k+1], d[i,k], d[i,k+1], x_tmp, K, k)

    @atomic y[i,j] = Base.ifelse(isoutside, x[i,j], yᵢⱼ) 
    @atomic LogJac[1, j] += Base.ifelse(isoutside, zero(typeof(LogJacᵢⱼ)), LogJacᵢⱼ)

    for par_ind in Base.OneTo(K-1)
        @atomic ∂y∂w[i,par_ind] += tangent[1][i,j] * Base.ifelse(isoutside, 0, ∂y∂wₖ[par_ind]) 
        @atomic ∂y∂h[i,par_ind] += tangent[1][i,j] * Base.ifelse(isoutside, 0, ∂y∂hₖ[par_ind]) 
        @atomic ∂LogJac∂w[i,par_ind] += tangent[2][1,j] * Base.ifelse(isoutside, 0, ∂LogJac∂wₖ[par_ind])
        @atomic ∂LogJac∂h[i,par_ind] += tangent[2][1,j] * Base.ifelse(isoutside, 0, ∂LogJac∂hₖ[par_ind])

        if par_ind <= K-2
            @atomic ∂y∂d[i,par_ind] += tangent[1][i,j] * Base.ifelse(isoutside, 0, ∂y∂dₖ[par_ind]) 
            @atomic ∂LogJac∂d[i,par_ind] += tangent[2][1,j] * Base.ifelse(isoutside, 0, ∂LogJac∂dₖ[par_ind])
        end
    end
end

function ChainRulesCore.rrule(
    ::typeof(spline_forward),
    x::AbstractArray{M0},
    w::AbstractArray{M1},
    h::AbstractArray{M2},
    d::AbstractArray{M3},
    w_logJac::AbstractArray{M4},
    h_logJac::AbstractArray{M5},
    d_logJac::AbstractArray{M6};
) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real, M6<:Real}

    # To do: Rewrite to avoid repeating calculation. 
    y, LogJac = spline_forward(x, w, h, d, w_logJac, h_logJac, d_logJac)
    pullback(tangent) = spline_forward_pullback(x, w, h, d, w_logJac, h_logJac, d_logJac, tangent)

    return (y, LogJac), pullback

end

function eval_forward_spline_params(
    wₖ::Real, wₖ₊₁::Real, 
    hₖ::Real, hₖ₊₁::Real, 
    dₖ::Real, dₖ₊₁::Real, 
    x::Real) 
      
    Δy = hₖ₊₁ - hₖ
    Δx = wₖ₊₁ - wₖ
    sk = Δy / Δx
    ξ = (x - wₖ) / Δx

    denom = (sk + (dₖ₊₁ + dₖ - 2*sk)*ξ*(1-ξ))
    nom_1 =  sk*ξ*ξ + dₖ*ξ*(1-ξ)
    nom_2 = Δy * nom_1
    nom_3 = dₖ₊₁*ξ*ξ + 2*sk*ξ*(1-ξ) + dₖ*(1-ξ)^2
    nom_4 = sk*sk*nom_3

    y = hₖ + nom_2/denom

    # LogJacobian
    LogJac = log(abs(nom_4))-2*log(abs(denom))

    return y, LogJac

end

function eval_forward_spline_params_with_grad(
    wₖ::M0, wₖ₊₁::M0, 
    hₖ::M1, hₖ₊₁::M1, 
    dₖ::M2, dₖ₊₁::M2, 
    x::M3, K::Integer, k::Integer) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}
      
    T = promote_type(M0, M1, M2, M3)

    ∂y∂w = zeros(T,K+1)
    ∂y∂h = zeros(T,K+1)
    ∂y∂d = zeros(T,K+1)

    ∂LogJac∂w = zeros(T,K+1)
    ∂LogJac∂h = zeros(T,K+1)
    ∂LogJac∂d = zeros(T,K+1)

    Δy = hₖ₊₁ - hₖ
    Δx = wₖ₊₁ - wₖ
    sk = Δy / Δx
    ξ = (x - wₖ) / Δx

    denom = (sk + (dₖ₊₁ + dₖ - 2*sk)*ξ*(1-ξ))
    nom_1 =  sk*ξ*ξ + dₖ*ξ*(1-ξ)
    nom_2 = Δy * nom_1
    nom_3 = dₖ₊₁*ξ*ξ + 2*sk*ξ*(1-ξ) + dₖ*(1-ξ)^2
    nom_4 = sk*sk*nom_3

    y = hₖ + nom_2/denom

    # LogJacobian
    LogJac = log(abs(nom_4))-2*log(abs(denom))

    # Gradient of parameters:

    # dy / dw_k
    ∂s∂wₖ = Δy/Δx^2
    ∂ξ∂wₖ = (-Δx + x - wₖ)/Δx^2
    ∂y∂wₖ = (Δy / denom^2) * ((∂s∂wₖ*ξ^2 + 2*sk*ξ*∂ξ∂wₖ + dₖ*(∂ξ∂wₖ -
                2*ξ*∂ξ∂wₖ))*denom - nom_1*(∂s∂wₖ - 2*∂s∂wₖ*ξ*(1-ξ) + (dₖ₊₁ + dₖ - 2*sk)*(∂ξ∂wₖ - 2*ξ*∂ξ∂wₖ)) )
    ∂LogJac∂wₖ = (1/nom_4)*(2*sk*∂s∂wₖ*nom_3 + sk*sk*(2*dₖ₊₁*ξ*∂ξ∂wₖ + 2*∂s∂wₖ*ξ*(1-ξ)+2*sk*(∂ξ∂wₖ - 2*ξ*∂ξ∂wₖ)-dₖ*2*(1-ξ)*∂ξ∂wₖ)) - (2/denom)*(∂s∂wₖ - 2*∂s∂wₖ*ξ*(1-ξ) + (dₖ₊₁ + dₖ - 2*sk)*(∂ξ∂wₖ - 2*ξ*∂ξ∂wₖ))

    # dy / dw_k+1
    ∂s∂wₖ₊₁ = -Δy/Δx^2
    ∂ξ∂wₖ₊₁ = -(x - wₖ) / Δx^2
    ∂y∂wₖ₊₁ = (Δy / denom^2) * ((∂s∂wₖ₊₁*ξ^2 + 2*sk*ξ*∂ξ∂wₖ₊₁ + dₖ*(∂ξ∂wₖ₊₁ -
                2*ξ*∂ξ∂wₖ₊₁))*denom - nom_1*(∂s∂wₖ₊₁ - 2*∂s∂wₖ₊₁*ξ*(1-ξ) + (dₖ₊₁ + dₖ - 2*sk)*(∂ξ∂wₖ₊₁ - 2*ξ*∂ξ∂wₖ₊₁)) )
    ∂LogJac∂wₖ₊₁ = (1/nom_4)*(2*sk*∂s∂wₖ₊₁*nom_3 + sk*sk*(2*dₖ₊₁*ξ*∂ξ∂wₖ₊₁ + 2*∂s∂wₖ₊₁*ξ*(1-ξ)+2*sk*(∂ξ∂wₖ₊₁ - 2*ξ*∂ξ∂wₖ₊₁)-dₖ*2*(1-ξ)*∂ξ∂wₖ₊₁)) - (2/denom)*(∂s∂wₖ₊₁ - 2*∂s∂wₖ₊₁*ξ*(1-ξ) + (dₖ₊₁ + dₖ - 2*sk)*(∂ξ∂wₖ₊₁ - 2*ξ*∂ξ∂wₖ₊₁))

    # dy / dh_k
    ∂s∂hₖ = -1/Δx
    ∂y∂hₖ = 1 + (1/denom^2)*((-nom_1+Δy*ξ*ξ*∂s∂hₖ)*denom - nom_2 * (∂s∂hₖ - 2*∂s∂hₖ*ξ*(1-ξ)) )
    ∂LogJac∂hₖ = (1/nom_4)*(2*sk*∂s∂hₖ*nom_3 + sk*sk*2*∂s∂hₖ*ξ*(1-ξ)) - (2/denom)*(∂s∂hₖ - 2*∂s∂hₖ*ξ*(1-ξ))

    # dy / dh_k+1
    ∂s∂hₖ₊₁ = 1/Δx
    ∂y∂hₖ₊₁ = (1/denom^2)*((nom_1+Δy*ξ*ξ*∂s∂hₖ₊₁)*denom - nom_2 * (∂s∂hₖ₊₁ - 2*∂s∂hₖ₊₁*ξ*(1-ξ)) )
    ∂LogJac∂hₖ₊₁ = (1/nom_4)*(2*sk*∂s∂hₖ₊₁*nom_3 + sk*sk*2*∂s∂hₖ₊₁*ξ*(1-ξ)) - (2/denom)*(∂s∂hₖ₊₁ - 2*∂s∂hₖ₊₁*ξ*(1-ξ))

    # dy / dd_k
    ∂y∂dₖ = (1/denom^2) * ((Δy*ξ*(1-ξ))*denom - nom_2*ξ*(1-ξ) )
    ∂LogJac∂dₖ = (1/nom_4)*sk^2*(1-ξ)^2 - (2/denom)*ξ*(1-ξ)

    # dy / dδ_k+1
    ∂y∂dₖ₊₁ = -(nom_2/denom^2) * ξ*(1-ξ)
    ∂LogJac∂dₖ₊₁ = (1/nom_4)*sk^2*ξ^2 - (2/denom)*ξ*(1-ξ)

    ∂y∂w[k+1] = ∂y∂wₖ₊₁
    ∂y∂h[k+1] = ∂y∂hₖ₊₁     
    ∂y∂d[k+1] = ∂y∂dₖ₊₁
    
    ∂y∂w[k] = ∂y∂wₖ
    ∂y∂h[k] = ∂y∂hₖ
    ∂y∂d[k] = ∂y∂dₖ

    ∂LogJac∂w[k+1] = ∂LogJac∂wₖ₊₁
    ∂LogJac∂h[k+1] = ∂LogJac∂hₖ₊₁
    ∂LogJac∂d[k+1] = ∂LogJac∂dₖ₊₁

    ∂LogJac∂w[k] = ∂LogJac∂wₖ
    ∂LogJac∂h[k] = ∂LogJac∂hₖ
    ∂LogJac∂d[k] = ∂LogJac∂dₖ

    return y, LogJac, ∂y∂w[2:end-1], ∂y∂h[2:end-1], ∂y∂d[2:end-2], ∂LogJac∂w[2:end-1], ∂LogJac∂h[2:end-1], ∂LogJac∂d[2:end-2]

end

# Transformation backward: 

function spline_backward(trafo::RationalQuadSplineInv, x::AbstractMatrix{<:Real})

    @assert size(trafo.widths, 1) == size(trafo.heights, 1) == size(trafo.derivatives, 1) == size(x, 1)  >= 1
    @assert size(trafo.widths, 2) == size(trafo.heights, 2) == (size(trafo.derivatives, 2) + 1)  >= 2

    w = cumsum_with_jac_mat(softmax_with_jac_mat(trafo.widths))
    h = cumsum_with_jac_mat(softmax_with_jac_mat(trafo.heights))
    d = softplus_with_jac_mat(trafo.derivatives)

    return spline_backward(x, w, h, d)
end


function spline_backward(
        x::AbstractArray{M0},
        w::AbstractArray{M1},
        h::AbstractArray{M2},
        d::AbstractArray{M3},
        B = 5.
    ) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

    T = promote_type(M0, M1, M2, M3)

    ndims = size(x, 1)
    nsmpls = size(x, 2)

    B = fill(B, size(w, 1))
    one = ones(size(d, 1))

    # add boundary conditions:
    w = convert(ElasticArray{eltype(w)}, w)
    h = convert(ElasticArray{eltype(h)}, h)
    d = convert(ElasticArray{eltype(d)}, d)

    prepend!(w, -B)
    prepend!(h, -B)
    prepend!(d, one)
    append!(d, one)

    y = zeros(T, ndims, nsmpls)
    LogJac = zeros(T, 1, nsmpls)

    device = KernelAbstractions.get_device(x)
    n = device isa GPU ? 256 : 4
    kernel! = spline_backward_kernel!(device, n)

    ev = kernel!(x, y, LogJac, w, h, d, ndrange=size(x))

    wait(ev)

    return y, LogJac
end

@kernel function spline_backward_kernel!(
        x::AbstractMatrix{M0},
        y::AbstractMatrix{M1},
        LogJac::AbstractMatrix{M2},
        w::AbstractMatrix{M3},
        h::AbstractMatrix{M4},
        d::AbstractMatrix{M5}
    ) where {M0<:Real, M1<:Real, M2<:Real, M3<:Real, M4<:Real, M5<:Real,}

    i, j = @index(Global, NTuple)
    
    K = size(w, 2)

    # Find the bin index
    k1 = searchsortedfirst_impl(h[i,:], x[i,j]) - 1
    k2 = one(typeof(k1))

    # Is outside of range
    isoutside = (k1 >= K) || (k1 == 0)
    k = Base.ifelse(isoutside, k2, k1)

    x_tmp = Base.ifelse(isoutside, h[i,k], x[i,j]) # Simplifies unnecessary calculations
    (yᵢⱼ, LogJacᵢⱼ) = eval_backward_spline_params(w[i,k], w[i,k+1], h[i,k], h[i,k+1], d[i,k], d[i,k+1], x_tmp)

    @atomic y[i,j] = Base.ifelse(isoutside, x[i,j], yᵢⱼ) 
    @atomic LogJac[1, j] += Base.ifelse(isoutside, zero(typeof(LogJacᵢⱼ)), LogJacᵢⱼ)

end

function eval_backward_spline_params(
    wₖ::M0, wₖ₊₁::M0, 
    hₖ::M1, hₖ₊₁::M1, 
    dₖ::M2, dₖ₊₁::M2, 
    x::M3) where {M0<:Real,M1<:Real, M2<:Real, M3<:Real}

    Δy = hₖ₊₁ - hₖ
    Δy2 = x - hₖ # use y instead of X, because of inverse
    Δx = wₖ₊₁ - wₖ
    sk = Δy / Δx

    a = Δy * (sk - dₖ) + Δy2 * (dₖ₊₁ + dₖ - 2*sk)
    b = Δy * dₖ - Δy2 * (dₖ₊₁ + dₖ - 2*sk)
    c = - sk * Δy2

    denom = -b - sqrt(b*b - 4*a*c)

    y = (2 * c / denom) * Δx + wₖ

    # Gradient computation:
    da =  (dₖ₊₁ + dₖ - 2*sk)
    db = -(dₖ₊₁ + dₖ - 2*sk)
    dc = - sk

    temp2 = 1 / (2*sqrt(b*b - 4*a*c))

    grad = 2 * dc * denom - 2 * c * (-db - temp2 * (2 * b * db - 4 * a * dc - 4 * c * da))
    LogJac = log(abs(Δx * grad)) - 2*log(abs(denom))

    return y, LogJac
end

# Utils: 

function softmax_with_jac(x::AbstractVector)

    exp_x = exp.(x)
    sum_exp_x = sum(exp_x)

    return exp_x ./ sum_exp_x 
end

function softmax_with_jac_mat(x::AbstractMatrix)

    val = cat([softmax_with_jac(i) for i in eachrow(x)]..., dims=2)'

    return val 
end

function cumsum_with_jac(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function cumsum_with_jac_mat(x::AbstractMatrix)

    return cat([cumsum_with_jac(i) for i in eachrow(x)]..., dims=2)'
end

function softplus_with_jac(x::AbstractVector)

    return log.(exp.(x) .+ 1) 
end

function softplus_with_jac_mat(x::AbstractMatrix)

    val = cat([softplus_with_jac(i) for i in eachrow(x)]..., dims=2)'

    return val
end

midpoint(lo::T, hi::T) where T<:Integer = lo + ((hi - lo) >>> 0x01)
binary_log(x::T) where {T<:Integer} = 8 * sizeof(T) - leading_zeros(x - 1)

function searchsortedfirst_impl(
        v::AbstractVector, 
        x::Real
    )
    
    u = one(Integer)
    lo = one(Integer) - u
    hi = length(v) + u
    
    n = binary_log(length(v))+1
    m = one(Integer)
    
    @inbounds for i in 1:n
        m_1 = midpoint(lo, hi)
        m = Base.ifelse(lo < hi - u, m_1, m)
        lo = Base.ifelse(v[m] < x, m, lo)
        hi = Base.ifelse(v[m] < x, hi, m)
    end
    return hi
end