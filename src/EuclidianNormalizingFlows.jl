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
using ElasticArrays
using ForwardDiffPullbacks
using Optim
using SpecialFunctions
using StatsBase
using ValueShapes

import ZygoteRules

include("householder_trafo.jl")
include("center_stretch.jl")

end # module
