# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


std_normal_logpdf(x::Real) = -(abs2(x) + log2π)/2


function mvnormal_ll_trafo(trafo::Function, X::AbstractMatrix{<:Real})
    Y = trafo(X)
    sum(std_normal_logpdf.(Y))
end


function mvnormal_ll_trafograd(trafo::Function, X::AbstractMatrix{<:Real})
    ll, back = Zygote.pullback(mvnormal_ll_trafo, trafo, X)
    d_trafo = back(one(eltype(X)))[1]
    return ll, d_trafo
end


function optimize_whitening(
    smpls::VectorOfSimilarVectors{<:Real}, initial_trafo::Function, optimizer;
    nbatches::Integer = 100, nepochs::Integer = 100,
    optstate = Optimisers.state(optimizer, deepcopy(initial_trafo)),
    ll_history = Vector{Float64}()
)
    batches = collect(Iterators.partition(smpls, nbatches))
    trafo::typeof(initial_trafo) = deepcopy(initial_trafo)
    state::typeof(optstate) = deepcopy(optstate)
    ll_hist = Vector{Float64}()
    for i in 1:nepochs
        for batch in batches
            X = flatview(batch)
            ll, d_trafo = mvnormal_ll_trafograd(trafo, X)
            neg_d_trafo = fmap(x -> -x, d_trafo) # Need to maximize likelihood
            state, trafo = Optimisers.update(optimizer, state, trafo, neg_d_trafo)
            push!(ll_hist, ll)
        end
    end
    (result = trafo, optimizer_state = state, ll_history = vcat(ll_history, ll_hist))
end
export optimize_trafo
