# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

using EuclidianNormalizingFlows
using Test

using EuclidianNormalizingFlows: center_stretch, center_contract


@testset "center_stretch" begin
    X = randn(10^3)

    @test @inferred(center_stretch(1f0, 7, 2, 4)) isa Float32
    @test @inferred(center_contract(1f0, 7, 2, 4)) isa Float32

    Y = center_stretch.(X, 7, 2, 4)
    X_reco = center_contract.(Y, 7, 2, 4)
    @test X ≈ X_reco
end
