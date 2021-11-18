# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

using EuclidianNormalizingFlows
using Test

using LinearAlgebra
using InverseFunctions, ChangesOfVariables
using ForwardDiff, Zygote

using EuclidianNormalizingFlows: householder_trafo, chained_householder_trafo, HouseholderTrafo


@testset "householder_trafo" begin
    v = rand(5)
    x = rand(5)
    X = rand(5, 3)
    
    householder_matrix(v::AbstractVector{<:Real}) = I - 2 * (v*v') / (v'*v)

    @test @inferred(householder_trafo(v, x)) ≈ householder_matrix(v) * x
    @test @inferred(householder_trafo(v, householder_trafo(v, x))) ≈ x

    @test @inferred(householder_trafo(v, X)) ≈ hcat(householder_trafo.(Ref(v), eachcol(X))...)
    @test @inferred(householder_trafo(v, X)) ≈ householder_matrix(v) * X
    @test @inferred(householder_trafo(v, householder_trafo(v, X))) ≈ X

    @test Zygote.pullback(householder_trafo, v, x)[1] == householder_trafo(v, x)
    @test Zygote.jacobian(householder_trafo, v, x)[1] ≈ ForwardDiff.jacobian(v -> householder_matrix(v) * x, v)
    @test Zygote.jacobian(householder_trafo, v, x)[2] ≈ ForwardDiff.jacobian(x -> householder_matrix(v) * x, x)

    @test Zygote.pullback(householder_trafo, v, X)[1] == householder_trafo(v, X)
    @test Zygote.jacobian(householder_trafo, v, X)[1] ≈ ForwardDiff.jacobian(v -> householder_matrix(v) * X, v)
    @test Zygote.jacobian(householder_trafo, v, X)[2] ≈ ForwardDiff.jacobian(X -> householder_matrix(v) * X, X)

    
    V = rand(5, 3)

    chained_householder_matrix(v::AbstractMatrix{<:Real}) = *(reverse(householder_matrix.(eachcol(V)))...)
    
    @test chained_householder_trafo(V, x) ≈ *(reverse(householder_matrix.(eachcol(V)))...) * x
    @test chained_householder_trafo(reverse(V, dims = 2), chained_householder_trafo(V, x)) ≈ x

    @test @inferred(chained_householder_trafo(V, X)) ≈ hcat(chained_householder_trafo.(Ref(V), eachcol(X))...)

    # # Yields incorrect results:
    # ForwardDiff.jacobian(V -> chained_householder_matrix(V) * x, V)
    # Zygote.jacobian(V -> chained_householder_matrix(V) * x, V)
    
    @test Zygote.pullback(chained_householder_trafo, V, x)[1] == chained_householder_trafo(V, x)
    @test Zygote.jacobian(chained_householder_trafo, V, x)[1] ≈ ForwardDiff.jacobian(V -> chained_householder_trafo(V, x), V)
    @test Zygote.jacobian(chained_householder_trafo, V, x)[2] ≈ ForwardDiff.jacobian(x -> chained_householder_trafo(V, x), x)

    @test Zygote.pullback(chained_householder_trafo, V, X)[1] == chained_householder_trafo(V, X)
    @test Zygote.jacobian(chained_householder_trafo, V, X)[1] ≈ ForwardDiff.jacobian(V -> chained_householder_trafo(V, X), V)
    @test Zygote.jacobian(chained_householder_trafo, V, X)[2] ≈ ForwardDiff.jacobian(X -> chained_householder_trafo(V, X), X)

    @test HouseholderTrafo(V) == HouseholderTrafo(V)
    @test isequal(HouseholderTrafo(V), HouseholderTrafo(V))
    @test hash(HouseholderTrafo(V)) == hash(HouseholderTrafo(V))

    for arg in (x, X)
        InverseFunctions.test_inverse(HouseholderTrafo(V), arg)
    end

    @test @inferred(with_logabsdet_jacobian(HouseholderTrafo(V), x)) == (chained_householder_trafo(V, x), 0)
    @test @inferred(with_logabsdet_jacobian(HouseholderTrafo(V), X)) == (chained_householder_trafo(V, X), fill(0, size(X, 2))')
end
