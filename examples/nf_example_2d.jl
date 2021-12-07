using ChangesOfVariables, InverseFunctions, ArraysOfArrays, Statistics
using Optimisers
using Plots

using EuclidianNormalizingFlows: ScaleShiftTrafo, CenterStretch, JohnsonTrafo, HouseholderTrafo
using EuclidianNormalizingFlows: optimize_whitening


XW = randn(2, 10^5)
histogram2d(XW[1,:], XW[2,:], nbins = 200, ratio = 1)

f_fwd_true =
    ScaleShiftTrafo([1.3, 0.4], [2.5, -1.2]) ∘
    HouseholderTrafo([1.0, 0.3]) ∘
    CenterStretch([4.0, 4.1], [2.0, 2.1], [3.0, 3.1]) # ∘
    # JohnsonTrafo([10.0, 11.0], [3.5, 3.6], [10.0, 11.0], [1.0, 1.1])
    
X = f_fwd_true(XW)
histogram2d(X[1,:], X[2,:], nbins = 200, ratio = 1)

initial_trafo =
    # inverse(JohnsonTrafo(rand(2), rand(2), rand(2), rand(2))) ∘
    inverse(CenterStretch([0.0, 0.0], [1.0, 1.0], [0.0, 0.0])) ∘
    inverse(HouseholderTrafo(randn(2))) ∘
    ScaleShiftTrafo([1.0, 1.0], [0.0, 0.0])

optimizer = ADAGrad()

smpls = nestedview(X)
nbatches = 1000
nepochs = 10

r = optimize_whitening(smpls, initial_trafo, optimizer, nbatches = nbatches, nepochs = nepochs)

XW2 = r.result(X)
histogram2d(XW2[1,:], XW2[2,:], nbins = 200)

cov(XW2, dims = 2)


using EuclidianNormalizingFlows: mvnormal_negll_trafo, mvnormal_negll_trafograd
batches = collect(Iterators.partition(smpls, round(Int, length(smpls)/nbatches)))
plot(r.negll_history, label = "batch negative log-likelihood")
ref_neg_ll = mean([mvnormal_negll_trafo(inverse(f_fwd_true), flatview(b)) for b in batches])
hline!([ref_neg_ll], label = "initial")
ref_neg_ll = mean([mvnormal_negll_trafo(initial_trafo, flatview(b)) for b in batches])
hline!([ref_neg_ll], label = "target")
