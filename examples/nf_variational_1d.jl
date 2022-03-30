


# Draw from source distribution (standard normal)
# calculate KL Divergence



using ChangesOfVariables, InverseFunctions, ArraysOfArrays, Statistics
using Optimisers
using Plots
using Distributions

using EuclidianNormalizingFlows: ScaleShiftTrafo, CenterStretch, JohnsonTrafo, HouseholderTrafo
using Zygote



std_normal_pdf(x::Real) = exp(-(abs2(x) + log(2*π))/2)

function my_ll(x)
    return log(0.3*std_normal_pdf.(x-2)+0.5*std_normal_pdf.(x-5) + 0.2*std_normal_pdf.(x+1))
end

function nELBO(trafo::Function, ξ::AbstractMatrix{<:Real})
    # P(x, z) joint distribution, x data, z model parameters
    # ξ ~ Q(ξ) = N(ξ|0,1) standard normal
    # z = f(ξ) with parametrized transformation f
    # ELBO = <ln(P(x, f(ξ)))>_Q(ξ) - <ln(Q(ξ))>_Q(ξ) + <log |df(ξ)/dξ|>_Q(ξ)

    z, ladj = with_logabsdet_jacobian(trafo, ξ)
    ξ_dim = size(ξ)[2] 
    N_samps =  size(ξ)[1]
    elbo = (sum(my_ll.(z)) + sum(ladj)) / N_samps - 0.5 * (log(2*π)+1)*ξ_dim
    return - elbo
end

function nELBO_trafograd(trafo::Function, ξ::AbstractMatrix{<:Real})
    nelbo, back = Zygote.pullback(nELBO, trafo, ξ)
    d_trafo = back(one(eltype(ξ)))[1]
    return nelbo, d_trafo
end

function optimise_ELBO(
    initial_trafo::Function, optimizer;
    batchsize::Integer = 100, nepochs::Integer = 100,
    optstate = Optimisers.setup(optimizer, deepcopy(initial_trafo)),
    nelbo_history = Vector{Float64}()
)
    trafo = deepcopy(initial_trafo)
    state = deepcopy(optstate)

    nelbo_hist = Vector{Float64}()
    for i in 1:nepochs
        batch = rand(Normal(),(batchsize,1))
        ξ_batch = flatview(batch)
        ξ_batch = vcat(ξ_batch,-ξ_batch) # antithetic sampling
        nelbo, d_trafo = nELBO_trafograd(trafo, ξ_batch)
        state, trafo = Optimisers.update(state, trafo, d_trafo)
        push!(nelbo_hist, nelbo)
    end
    (result = trafo, optimizer_state = state, nelbo_history = vcat(nelbo_history, nelbo_hist))
end


initial_trafo_fwd =
    JohnsonTrafo([0.0], [5.0], [0.0], [5.0]) ∘
    inverse(CenterStretch([0.0], [1.0], [0.0])) ∘
    JohnsonTrafo([0.0], [5.0], [0.0], [5.0]) ∘
    inverse(CenterStretch([0.0], [1.0], [0.0]))

 
initial_trafo = inverse(initial_trafo_fwd)


optimizer = ADAGrad()

r = optimise_ELBO(initial_trafo, optimizer, batchsize=100, nepochs=1000)


my_ξ_samps = rand(Normal(),100000)

my_z_samps = r.result(my_ξ_samps)

x = collect(-5:0.1:10)

stephist(my_z_samps, nbins = 100; normalize = true, label="approximation")
plot!(x, exp.(my_ll.(x)), label="ground truth")

plot(r.nelbo_history, label = "batch nELBO")


