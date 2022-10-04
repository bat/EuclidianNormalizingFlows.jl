# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

function get_flow(n_dims::Integer)
    trafos = Function[]

    d = floor(Int, n_dims/2)

    for i in 1:d
        mask = 

    end


    return fchain(trafos)
end 


function _get_nns(n_dims::Integer, K::Integer, hidden::Integer)
    d = floor(Int, n_dims/2)
    nns = Chain[]

    for i in 1:(n_dims-d)
        nn_tmp = Chain(Dense(d => hidden, relu),
                       Dense(hidden => hidden, relu),
                       Dense(hidden => 3K-1)
        )
        push!(nns, nn_tmp)
    end

    return nns
end

function get_params(nns::AbstractArray, x::AbstractMatrix)
    
    res = format_params(nns[1](x[:,1]))
    for i in 2:length(nns)
        res = hcat(res,format_params(nns[i](x[:,1])))
    end

    for j in 2:size(x,2)
        res_tmp = format_params(nns[1](x[:,j]))
        for k in 2:length(nns)
            res_tmp = hcat(res_tmp,format_params(nns[k](x[:,j])))
        end
        res = cat(res,res_tmp,dims=3)
    end

    return permutedims(res,(2,3,1))
end

export get_params

function format_params(θ::AbstractVector)
    K = Int((length(θ)+1)/3)
    res = vcat(-5,_cumsum(_softmax(θ[1:K])))
    res = vcat(res,-5,_cumsum(_softmax(θ[K+1:2K])))
    res = vcat(res,1,_softmax(θ[2K+1:end]),1)

    return res
end

export format_params


function _softmax(x::AbstractVector)

    exp_x = exp.(x)
    sum_exp_x = sum(exp_x)

    return exp_x ./ sum_exp_x 
end

function _softmax(x::AbstractMatrix)

    val = cat([_softmax(i) for i in eachrow(x)]..., dims=2)'

    return val 
end

function _cumsum(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function _cumsum(x::AbstractMatrix)

    return cat([_cumsum(i) for i in eachrow(x)]..., dims=2)'
end

function _softplus(x::AbstractVector)

    return log.(exp.(x) .+ 1) 
end

function _softplus(x::AbstractMatrix)

    val = cat([_softplus(i) for i in eachrow(x)]..., dims=2)'

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
