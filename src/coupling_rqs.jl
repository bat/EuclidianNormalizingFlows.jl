
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