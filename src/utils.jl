# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

function get_flow(n_dims::Integer, K::Integer=10, hidden::Integer=20)
    d = floor(Int, n_dims/2) 
    i = 1
    all_dims = [1:n_dims...]
    trafos = Function[]
    
    while d <= n_dims
        mask1 = [i:d...]
        mask2 = all_dims[.![(el in all_dims && el in mask1) for el in all_dims]]
        nns1 = _get_nns(n_dims, K, hidden)
        nns2 = _get_nns(n_dims, K, hidden)
        
        d+=1
        i+=1

        push!(trafos, CouplingRQS(nns1, mask1, mask2))
        push!(trafos, CouplingRQS(nns2, mask2, mask1))
    end

    return fchain(trafos)
end 

export get_flow

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
    
    res = nns[1](x[:,1])

    for i in 2:length(nns)
        res = hcat(res,nns[i](x[:,1]))
    end

    for j in 2:size(x,2)
        res_tmp = nns[1](x[:,j])
        for k in 2:length(nns)
            res_tmp = hcat(res_tmp,nns[k](x[:,j]))
        end
        res = cat(res,res_tmp,dims=3)
    end

    res = permutedims(res,(2,3,1))

    K = Int((size(res,3)+1)/3)

    return format_params(res[:,:,1:K], res[:,:,K+1:2K], res[:,:,2K+1:end])
end

export get_params


function format_params(raw_w::AbstractArray, raw_h::AbstractArray, raw_d::AbstractArray)

    one_pad = repeat([1], size(raw_w, 1), size(raw_w, 2))
    five_pad = repeat([-5], size(raw_w, 1), size(raw_w, 2))

    w = cat(ignore_derivatives(five_pad),_cumsum_tri(_softmax_tri(raw_w)),dims=3)
    h = cat(ignore_derivatives(five_pad),_cumsum_tri(_softmax_tri(raw_h)),dims=3)
    d = cat(ignore_derivatives(one_pad),_softmax_tri(raw_d),ignore_derivatives(one_pad),dims=3)

    return w,h,d
end

export format_params


function _sort_dimensions(x::AbstractMatrix, y::AbstractMatrix, mask1::AbstractVector)
    
    if 1 in mask1
        res = reshape(x[1,:],1,size(x,2))
        c1 = 2
        c2 = 1
    else
        res = reshape(y[1,:],1,size(x,2))
        c1 = 1
        c2 = 2
    end

    for i in 2:(size(x,1)+size(y,1))
        if i in mask1
            res = vcat(res, reshape(x[c1,:],1,size(x,2)))
            c1+=1
        else
            res = vcat(res, reshape(y[c2,:],1,size(y,2)))
            c2+=1
        end
    end
    
    return res
end


function _softmax(x::AbstractVector)

    exp_x = exp.(x)
    sum_exp_x = sum(exp_x)

    return exp_x ./ sum_exp_x 
end

function _softmax(x::AbstractMatrix)

    val = cat([_softmax(i) for i in eachrow(x)]..., dims=2)'

    return val 
end

function _softmax_tri(x::AbstractArray)
    slices = [slice for slice in eachslice(x,dims=2)]
    res = _softmax(slices[1])

    for i in 2:length(slices)
        res = cat(res, _softmax(slices[i]),dims=3)
    end
    return permutedims(res, (1,3,2))
end

function _cumsum(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function _cumsum(x::AbstractMatrix)

    return cat([_cumsum(i) for i in eachrow(x)]..., dims=2)'
end

function _cumsum_tri(x::AbstractArray)
    slices = [slice for slice in eachslice(x,dims=2)]
    res = _cumsum(slices[1])

    for i in 2:length(slices)
        res = cat(res, _cumsum(slices[i]),dims=3)
    end
    return permutedims(res, (1,3,2))
end


function _softplus(x::AbstractVector)

    return log.(exp.(x) .+ 1) 
end

function _softplus(x::AbstractMatrix)

    val = cat([_softplus(i) for i in eachrow(x)]..., dims=2)'

    return val
end

function _softplus_tri(x::AbstractArray)
    return log.(exp.(x) .+ 1) 
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


# function get_params(nns::AbstractArray, x::AbstractMatrix)
    
#     res = format_params(nns[1](x[:,1]))

#     for i in 2:length(nns)
#         res = hcat(res,format_params(nns[i](x[:,1])))
#     end

#     for j in 2:size(x,2)
#         res_tmp = format_params(nns[1](x[:,j]))
#         for k in 2:length(nns)
#             res_tmp = hcat(res_tmp,format_params(nns[k](x[:,j])))
#         end
#         res = cat(res,res_tmp,dims=3)
#     end

#     res = permutedims(res,(2,3,1))

#     K = Int((size(res,3)-3)/3)
    
#     w = res[:,:,1:K+1]
#     h = res[:,:,K+2:2(K+1)]
#     d = res[:,:,2(K+1)+1:end]

#     return w, h, d 
# end