# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

@testset "test_forward_inverse_are_consistent_constraint" begin

    num_bins = 30

    unnormalized_widths      = abs.(randn(Float64, num_bins))
    unnormalized_heights     = abs.(randn(Float64, num_bins))
    unnormalized_derivatives = abs.(randn(Float64, num_bins + 1))
    lambdas                  = abs.(randn(Float64, num_bins))

    function call_spline_fn(inputs, inverse)
        return rational_linear_spline(

            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            lambdas,
            inverse
        )
    end 

    inputs = rand(Float64, num_bins)
    outputs, logabsdet = call_spline_fn(inputs, false)
    inputs_inv, logabsdet_inv = call_spline_fn(outputs, true)

    eps = 1e-12
    @test inputs ≈ inputs_inv
    @test isapprox(logabsdet + logabsdet_inv, zero(logabsdet); atol = eps)
end

@testset "test_forward_inverse_are_consistent_unconstraint" begin

    num_bins = 10

    unnormalized_widths      = abs.(randn(Float64, num_bins))
    unnormalized_heights     = abs.(randn(Float64, num_bins))
    unnormalized_derivatives = abs.(randn(Float64, num_bins))
    lambdas                  = abs.(randn(Float64, num_bins))

    function call_spline_fn(inputs, inverse)
        return unconstrained_rational_linear_spline(
            inputs,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            lambdas,
            inverse
        )
        
    end 

    inputs = 3 * randn(Float64, num_bins) # Note inputs are outside [0,1]
    outputs, logabsdet = call_spline_fn(inputs, false)
    inputs_inv, logabsdet_inv = call_spline_fn(outputs, true)

    eps = 1e-15

    @test inputs ≈ inputs_inv
    @test isapprox(logabsdet + logabsdet_inv, zero(logabsdet); atol = eps)
end
