# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

__precompile__(true)

"""
    EuclidianNormalizingFlows

Euclidian normalizing flows.
"""
module EuclidianNormalizingFlows

using LinearAlgebra
using Random
using Statistics

using ArgCheck
using ArraysOfArrays
using ChainRulesCore
using Distributions
using DocStringExtensions
using ElasticArrays
using ForwardDiffPullbacks
using Functors
using Optim
using Optimisers
using Parameters
using SpecialFunctions
using StatsBase
using ValueShapes

import ZygoteRules

using Distributions: log2π


include("abstract_trafo.jl")
include("optimize_whitening.jl")
include("householder_trafo.jl")
include("center_stretch.jl")
include("johnson_trafo.jl")

end # module
