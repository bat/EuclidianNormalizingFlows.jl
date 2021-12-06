using ChangesOfVariables, InverseFunctions, ArraysOfArrays, Statistics
using Optimisers
using Plots

using EuclidianNormalizingFlows: CenterStretch, JohnsonTrafo, HouseholderTrafo
using EuclidianNormalizingFlows: optimize_whitening

f_fwd_true =
    CenterStretch([4.0], [1.0], [0.0]) ∘
    JohnsonTrafo([10.0], [3.5], [10.0], [1.0])
    
XW = randn(1, 10^5)
stephist(XW[1,:], nbins = 100; normalize = true)

X = f_fwd_true(XW)
stephist!(X[1,:], nbins = 100; normalize = true)
    

initial_trafo =
    JohnsonTrafo([0.0], [5.0], [0.0], [5.0]) ∘
    inverse(CenterStretch([0.0], [1.0], [0.0])) ∘
    JohnsonTrafo([0.0], [5.0], [0.0], [5.0]) ∘
    inverse(CenterStretch([0.0], [1.0], [0.0]))

optimizer = ADAGrad()

smpls = nestedview(X)
nbatches = 100
nepochs = 10

r = optimize_whitening(smpls, initial_trafo, optimizer, nbatches = nbatches, nepochs = nepochs)

XW2 = r.result(X)
stephist(XW[1,:], nbins = 100; normalize = true)
stephist!(XW2[1,:], nbins = 100; normalize = true)


using EuclidianNormalizingFlows: mvnormal_negll_trafo, mvnormal_negll_trafograd
batches = collect(Iterators.partition(smpls, round(Int, length(smpls)/nbatches)))
plot(r.negll_history, label = "batch negative log-likelihood")
ref_neg_ll = mean([mvnormal_negll_trafo(inverse(f_fwd_true), flatview(b)) for b in batches])
hline!([ref_neg_ll], label = "initial")
ref_neg_ll = mean([mvnormal_negll_trafo(initial_trafo, flatview(b)) for b in batches])
hline!([ref_neg_ll], label = "target")
