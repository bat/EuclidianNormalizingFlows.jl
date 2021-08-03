# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

using EuclidianNormalizingFlows
using Test

using LinearAlgebra
using ForwardDiff, Zygote

using EuclidianNormalizingFlows: householder_trafo, chained_householder_trafo


@testset "householder_trafo" begin
    v = rand(5)
    x = rand(5)
    
    householder_matrix(v::AbstractVector{<:Real}) = I - 2 * (v*v') / (v'*v)
    
    @test householder_trafo(v, x) ≈ householder_matrix(v) * x
    @test householder_trafo(v, householder_trafo(v, x)) ≈ x
    
    @test Zygote.pullback(householder_trafo, v, x)[1] == householder_trafo(v, x)
    @test Zygote.jacobian(householder_trafo, v, x)[1] ≈ ForwardDiff.jacobian(v -> householder_matrix(v) * x, v)
    @test Zygote.jacobian(householder_trafo, v, x)[2] ≈ ForwardDiff.jacobian(x -> householder_matrix(v) * x, x)
    
    
    V = rand(5, 3)
    
    chained_householder_matrix(v::AbstractMatrix{<:Real}) = *(reverse(householder_matrix.(eachcol(V)))...)
    
    @test chained_householder_trafo(V, x) ≈ *(reverse(householder_matrix.(eachcol(V)))...) * x
    @test chained_householder_trafo(reverse(V, dims = 2), chained_householder_trafo(V, x)) ≈ x
    
    # # Yields incorrect results:
    # ForwardDiff.jacobian(V -> chained_householder_matrix(V) * x, V)
    # Zygote.jacobian(V -> chained_householder_matrix(V) * x, V)
    
    @test Zygote.pullback(chained_householder_trafo, V, x)[1] == chained_householder_trafo(V, x)
    @test Zygote.jacobian(chained_householder_trafo, V, x)[1] ≈ ForwardDiff.jacobian(V -> chained_householder_trafo(V, x), V)
    @test Zygote.jacobian(chained_householder_trafo, V, x)[2] ≈ ForwardDiff.jacobian(x -> chained_householder_trafo(V, x), x)
end
