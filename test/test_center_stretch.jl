# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

using EuclidianNormalizingFlows
using Test

using InverseFunctions, ChangesOfVariables
using ForwardDiff

using EuclidianNormalizingFlows: center_stretch, center_contract, center_contract_ladj, CenterStretch, CenterContract


@testset "center_stretch" begin
    X = randn(10^3)

    @test @inferred(center_stretch(1f0, 7, 2, 4)) isa Float32
    @test @inferred(center_contract(1f0, 7, 2, 4)) isa Float32

    @test center_stretch(1f0, 7, 2, 4) ≈ 11.927293f0
    @test center_contract(12f0, 7, 2, 4) ≈ 1.063464f0

    Y = center_stretch.(X, 7, 2, 4)
    X_reco = center_contract.(Y, 7, 2, 4)
    @test X ≈ X_reco

    @test isapprox(@inferred(center_contract_ladj(4.2, 4, 2, 3)), log(abs(ForwardDiff.derivative(x -> center_contract(x, 4, 2, 3), 4.2))), rtol = 0.01)
    @test isapprox(@inferred(-center_contract_ladj(4.2, 4, 2, 3)), log(abs(ForwardDiff.derivative(x -> center_stretch(x, 4, 2, 3), center_contract(4.2, 4, 2, 3)))), rtol = 0.01)

    @inferred(CenterStretch(4, 2, 3)) isa CenterStretch
    trafo = CenterStretch(4, 2, 3)

    @test @inferred(trafo(4.2)) == center_stretch(4.2, 4, 2, 3)

    @test @inferred(with_logabsdet_jacobian(CenterStretch([4.0, 4.1], [2.0, 2.1], [3.0, 3.1]), [4.2, 4.3])) == (
        center_stretch.([4.2, 4.3], [4.0, 4.1], [2.0, 2.1], [3.0, 3.1]),
        sum(-center_contract_ladj.(center_stretch.([4.2, 4.3], [4.0, 4.1], [2.0, 2.1], [3.0, 3.1]), [4.0, 4.1], [2.0, 2.1], [3.0, 3.1])),
    )

    @test @inferred(with_logabsdet_jacobian(CenterContract([4.0, 4.1], [2.0, 2.1], [3.0, 3.1]), [11.0, 11.5])) == (
        center_contract.([11.0, 11.5], [4.0, 4.1], [2.0, 2.1], [3.0, 3.1]),
        sum(center_contract_ladj.([11.0, 11.5], [4.0, 4.1], [2.0, 2.1], [3.0, 3.1])),
    )

    let
        fwd = CenterStretch([4.0, 4.1], [2.0, 2.1], [3.0, 3.1])
        @test fwd == deepcopy(fwd)
        @test isequal(fwd, deepcopy(fwd))
        @test hash(fwd) == hash(deepcopy(fwd))
    
        rev = inverse(fwd)
        @test rev == deepcopy(rev)
        @test isequal(rev, deepcopy(rev))
        @test hash(rev) == hash(deepcopy(rev))

        X = randn(2, 3)
        InverseFunctions.test_inverse(fwd, X)

        Y, ladjs = with_logabsdet_jacobian(fwd, X)
        @test hcat((getindex).(with_logabsdet_jacobian.(fwd, eachcol(X)), 1)...) == Y
        @test (getindex).(with_logabsdet_jacobian.(fwd, eachcol(X)), 2) == ladjs
        X2, inv_ladjs = with_logabsdet_jacobian(rev, Y)
        @test X2 ≈ X
        @test inv_ladjs ≈ - ladjs
    end
end
