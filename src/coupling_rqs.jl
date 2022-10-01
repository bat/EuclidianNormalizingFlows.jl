
struct CouplingRQS <: Function
    nns::AbstractArray
end

function CouplingRQS(n_dims::Integer, K::Integer=10, hidden::Integer=20)

    return CouplingRQS(_get_nns(n_dims, K, hidden))
end


function _get_nns(n_dims::Integer, K::Integer, hidden::Integer)
    d = floor(Int, n_dims/2)
    nns = Chain[]

    for i in 1:d
        nn_tmp = Chain(Dense(d => hidden)
            Dense(hidden => hidden)
            Dense(hidden => 3K-1)
        )
        append!(nns, nn_tmp)
    end

    return nns
end

#=


function _get_nns(n_dims::Integer, K::Integer = 20)

    d = n_dims > 2 ? round(Integer, n_dims / 2) : 1

    nn1 = Chain(Dense(d => 20, relu),
                Dense(20 => 20, relu),
                Dense(20 => (3K-1))
                )

    nn2 = Chain(Dense((n_dims - d) => 20, relu),
                Dense(20 => 20, relu),
                Dense(20 => (3K-1))
                )
    
    return nn1, nn2
end

function get_weights(n_dims::Integer)

    d = n_dims > 2 ? round(Integer, n_dims / 2) : 1

    w1 = Flux.glorot_uniform(20, d)
    w2 = Flux.glorot_uniform(20, d)

    return w1, w2
end

export get_weights

function get_params(θ::AbstractMatrix, N::Integer, K::Integer)

    w = _cumsum(_softmax(θ[1:K,:]))
    h = _cumsum(_softmax(θ[K+1:2K,:]))
    d = _softplus(θ[2K+1:end,:])

    w = vcat(repeat([-5,], 1, N), w)
    h = vcat(repeat([-5,], 1, N), h)
    d = vcat(repeat([1,], 1, N), d)
    d = vcat(d, repeat([1,], 1, N))

    return w, h, d
end

function _softmax(x::AbstractVector)
    exp_x = exp.(x)
    sum_exp_x = sum(exp_x)
    return exp_x ./ sum_exp_x 
end

function _softmax(x::AbstractMatrix)
    return hcat(_softmax.(x[:,i] for i in axes(x,2))...)
end

function _cumsum(x::AbstractVector; B = 5)
    return 2 .* B .* cumsum(x) .- B 
end

function _cumsum(x::AbstractMatrix; B = 5)
    return copy((2 .* B .* cumsum(x, dims = 1) .- B))
end

function _softplus(x::AbstractVector)
    return log.(exp.(x) .+ 1) 
end

function _softplus(x::AbstractMatrix)
    return (log.(exp.(x) .+ 1))
end

function _sigmoid(x::Real)
    return 1 / (1 + exp(-x))
end

function _relu(x::Real)
    return x > 0 ? x : 0
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

=#