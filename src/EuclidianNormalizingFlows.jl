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
using ChangesOfVariables
using Distributions
using DocStringExtensions
using ElasticArrays
using ForwardDiffPullbacks
using Functors
using InverseFunctions
using Optim
using Optimisers
using Parameters
using SpecialFunctions
using StatsBase
using ValueShapes
using KernelAbstractions

import Zygote
import ZygoteRules

using Distributions: log2π

import InverseFunctions.inverse
import ChainRulesCore.rrule
import ChangesOfVariables.with_logabsdet_jacobian


include("abstract_trafo.jl")
include("optimize_whitening.jl")
include("householder_trafo.jl")
include("scale_shift_trafo.jl")
include("center_stretch.jl")
include("johnson_trafo.jl")
include("spline_trafo.jl")

end # module
