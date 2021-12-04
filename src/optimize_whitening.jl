# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


std_normal_logpdf(x::Real) = -(abs2(x) + log2π)/2


function mvnormal_negll_trafo(trafo::Function, X::AbstractMatrix{<:Real})
    nsamples = size(X, 2) # normalize by number of samples to be independent of batch size:
    Y, ladj = with_logabsdet_jacobian(trafo, X)
    #ref_ll = sum(sum(std_normal_logpdf.(Y), dims = 1) .+ ladj) / nsamples
    # Faster:
    ll = (sum(std_normal_logpdf.(Y)) + sum(ladj)) / nsamples
    #@assert ref_ll ≈ ll
    return -ll
end


function mvnormal_negll_trafograd(trafo::Function, X::AbstractMatrix{<:Real})
    ll, back = Zygote.pullback(mvnormal_negll_trafo, trafo, X)
    d_trafo = back(one(eltype(X)))[1]
    return ll, d_trafo
end


function optimize_whitening(
    smpls::VectorOfSimilarVectors{<:Real}, initial_trafo::Function, optimizer;
    nbatches::Integer = 100, nepochs::Integer = 100,
    optstate = Optimisers.state(optimizer, deepcopy(initial_trafo)),
    ll_history = Vector{Float64}()
)
    batchsize = round(Int, length(smpls) / nbatches)
    batches = collect(Iterators.partition(smpls, batchsize))
    trafo = deepcopy(initial_trafo)
    state = deepcopy(optstate)
    ll_hist = Vector{Float64}()
    for i in 1:nepochs
        for batch in batches
            X = flatview(batch)
            ll, d_trafo = mvnormal_negll_trafograd(trafo, X)
            state, trafo = Optimisers.update(optimizer, state, trafo, d_trafo)
            push!(ll_hist, ll)
        end
    end
    (result = trafo, optimizer_state = state, ll_history = vcat(ll_history, ll_hist))
end
export optimize_trafo
