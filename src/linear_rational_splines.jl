# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

const DEFAULT_MIN_BIN_WIDTH = 1e-3
const DEFAULT_MIN_BIN_HEIGHT = 1e-3
const DEFAULT_MIN_DERIVATIVE = 1e-3

"""
    unconstrained_rational_linear_spline(   inputs::AbstractVector,
                                            unnormalized_widths::AbstractVector,
                                            unnormalized_heights::AbstractVector,
                                            unnormalized_derivatives::AbstractVector,
                                            unnormalized_lambdas::AbstractVector,
                                            inverse::Bool = false,
                                            tail_bound::AbstractFloat = 1.,
                                            min_bin_width::AbstractFloat = DEFAULT_MIN_BIN_WIDTH,
                                            min_bin_height::AbstractFloat = DEFAULT_MIN_BIN_HEIGHT,
                                            min_derivative::AbstractFloat = DEFAULT_MIN_DERIVATIVE)

    Evaluate the linear rational spline transformation g(x) of input data x for inputs that are not 
    constraint to the interval mask. Inputs that lie outside the interval mask are transformed with 
    the identity transformation.

    *inputs*, *unnormalized_widths*, *unnormalized_heights*, *unnormalized_derivatives*, 
    and *unnormalized_lambdas* must all have the same lenght
"""
function unconstrained_rational_linear_spline(  inputs::AbstractVector,
                                                unnormalized_widths::AbstractVector,
                                                unnormalized_heights::AbstractVector,
                                                unnormalized_derivatives::AbstractVector,
                                                unnormalized_lambdas::AbstractVector,
                                                inverse::Bool = false,
                                                tail_bound::AbstractFloat = 1.,
                                                min_bin_width::AbstractFloat = DEFAULT_MIN_BIN_WIDTH,
                                                min_bin_height::AbstractFloat = DEFAULT_MIN_BIN_HEIGHT,
                                                min_derivative::AbstractFloat = DEFAULT_MIN_DERIVATIVE)

    # find indices of inputs that lie inside and outside the interval mask
    inside_interval_mask = (inputs .>= -tail_bound) .& (inputs .<= tail_bound)
    outside_interval_mask = .~inside_interval_mask

    # intitiate return values
    outputs   = zero(inputs)
    logabsdet = zero(inputs)

    # create workable derivative vector for rational_linear_spline(); padded with constant values at the interval ends
    unnorm_deriv_inside_interval_mask = unnormalized_derivatives[inside_interval_mask]
    constant = log(exp(1 - min_derivative) - 1)
    pushfirst!(unnorm_deriv_inside_interval_mask, constant)
    unnorm_deriv_inside_interval_mask[end] =  constant

    # "apply" identity transformation to inputs outside the interval mask
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] .= 0

    # apply linear rational spline transform to inputs inside the interval mask
    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_linear_spline(
                                inputs[inside_interval_mask],
                                unnormalized_widths[inside_interval_mask],
                                unnormalized_heights[inside_interval_mask],
                                unnorm_deriv_inside_interval_mask,
                                unnormalized_lambdas[inside_interval_mask],
                                inverse,
                                -tail_bound, 
                                tail_bound, 
                                -tail_bound, 
                                tail_bound,
                                min_bin_width,
                                min_bin_height,
                                min_derivative)


    return outputs, logabsdet
end 

"""
    rational_linear_spline( inputs::AbstractVector,
                            unnormalized_widths::AbstractVector,
                            unnormalized_heights::AbstractVector,
                            unnormalized_derivatives::AbstractVector,
                            unnormalized_lambdas::AbstractVector,
                            inverse::Bool = false,
                            left::AbstractFloat = 0., 
                            right::AbstractFloat = 1., 
                            bottom::AbstractFloat = 0., 
                            top::AbstractFloat = 1.,
                            min_bin_width::AbstractFloat = DEFAULT_MIN_BIN_WIDTH,
                            min_bin_height::AbstractFloat = DEFAULT_MIN_BIN_HEIGHT,
                            min_derivative::AbstractFloat = DEFAULT_MIN_DERIVATIVE)


    Evaluate the linear rational spline transformation g(x) of input data x for inputs constraint 
    to the interval mask. 

    *inputs*, *unnormalized_widths*, *unnormalized_heights*, and *unnormalized_lambdas* must all 
    have the same lenght.

    *unnormalized_derivatives* must of length (lenght(inputs) + 1) 
"""
function rational_linear_spline(inputs::AbstractVector,
                                unnormalized_widths::AbstractVector,
                                unnormalized_heights::AbstractVector,
                                unnormalized_derivatives::AbstractVector,
                                unnormalized_lambdas::AbstractVector,
                                inverse::Bool = false,
                                left::AbstractFloat = 0., 
                                right::AbstractFloat = 1., 
                                bottom::AbstractFloat = 0., 
                                top::AbstractFloat = 1.,
                                min_bin_width::AbstractFloat = DEFAULT_MIN_BIN_WIDTH,
                                min_bin_height::AbstractFloat = DEFAULT_MIN_BIN_HEIGHT,
                                min_derivative::AbstractFloat = DEFAULT_MIN_DERIVATIVE)

    if (minimum(inputs) < left) | (maximum(inputs) > right)
        Error("Input Outside Domain")
    end 

    num_bins = length(unnormalized_widths)

    if (min_bin_width * num_bins > 1.0)
        Error("Minimal bin width too large for the number of bins")
    end 

    if (min_bin_height * num_bins > 1.0)
        Error("Minimal bin height too large for the number of bins")
    end 

    # interpret the learned parameters as bin widths:
    widths = softmax(unnormalized_widths)  
    widths = min_bin_width .+ (1 - min_bin_width * num_bins) * widths

    # calculate the bin edges ( = x values of the knots), stored in cumwidhts:
    cumwidths = cumsum(widths)   
    cumwidths = (right - left) * cumwidths .+ left

    pushfirst!(cumwidths, left) # add the edge of the left most bin (left end of the interval mask)
    cumwidths[end] = right      # ensure that the edge of the right most bin is the correct value (right end of the interval mask) (maybe unnecessary)

    widths = cumwidths[2:end] .- cumwidths[1:end - 1]

    # interpret the learned parameters as the derivatives of the spline functions at their respective knot:
    derivatives = min_derivative .+ softplus.(unnormalized_derivatives)

    # interpret the learned parameters as bin heights
    heights = softmax(unnormalized_heights)
    heights = min_bin_height .+ (1 - min_bin_height * num_bins) * heights

    # calculate the upper bin edges ( = y values of the knots), stored in cumheights:
    cumheights = cumsum(heights)          
    cumheights = (top - bottom) * cumheights .+ bottom

    pushfirst!(cumheights, bottom) # add the height of the left most bin (bottom end of the interval mask)
    cumheights[end] = top          # ensure that the upper edge of the right most bin is the correct value (top end of the interval mask) (maybe unnecessary)

    heights = cumheights[2:end] .- cumheights[1:end - 1]

    # to evaluate a spline transform at location x, find in which bin x lies:
    if inverse
        bin_idx = searchsortedlast.(Ref(cumheights), inputs)
    else
        bin_idx = searchsortedlast.(Ref(cumwidths), inputs)
    end 

    # pick out the spline prameters that corresponds to the input data:
    input_cumwidths = cumwidths[bin_idx]
    input_bin_widths = widths[bin_idx]

    input_cumheights = cumheights[bin_idx]
    input_heights = heights[bin_idx]

    delta = heights ./ widths
    input_delta = delta[bin_idx]

    input_derivatives = derivatives[1:end - 1][bin_idx]      # kth derivative 
    input_derivatives_plus_one = derivatives[2:end][bin_idx] # k + 1st derivative

    # define the lambdas for the bins (see 'Invertible Generative Modeling using Linear Rational Splines' Dolatabadi et al., sec.3.2)
    lambdas = 0.95 * sigmoid.(unnormalized_lambdas) .+ 0.025

    # define spline function prameters:
    lam = lambdas[bin_idx]
    wa  = 1
    wb  = sqrt.(input_derivatives./input_derivatives_plus_one) * wa
    wc  = (lam * wa .* input_derivatives .+ (1 .- lam) .* wb .* input_derivatives_plus_one)./input_delta
    ya  = input_cumheights
    yb  = input_heights .+ input_cumheights
    yc  = ((1 .- lam) * wa .* ya .+ lam .* wb .* yb)./ ((1 .- lam) * wa .+ lam .* wb)

    # compute spline functions and store values in 'outputs' aswell as the log abs value of the determinant in 'logabsdet'
    # the boolean factors in the computation determine in which 'virtual bin' the input x lies (see 'Invertible Generative Modeling using Linear Rational Splines' Dolatabadi et al., sec.3.2)
    if inverse

        numerator = (lam * wa .* (ya .- inputs)) .* float(inputs .<= yc) .+ ((wc .- lam .* wb) .* inputs .+ lam .* wb .* yb .- wc .* yc) .* float(inputs .> yc)

        denominator = ((wc .- wa) .* inputs .+ wa * ya .- wc .* yc) .* float(inputs .<= yc) .+ ((wc .- wb) .* inputs .+ wb .* yb .- wc .* yc) .* float(inputs .> yc)

        theta = numerator./denominator

        outputs = theta .* input_bin_widths .+ input_cumwidths

        derivative_numerator = (wa * wc .* lam .* (yc .- ya) .* float(inputs .<= yc) .+ wb .* wc .* (1 .- lam) .* (yb .- yc) .* float(inputs .> yc)) .* input_bin_widths

        logabsdet = log.(derivative_numerator) .- 2 * log.(abs.(denominator))

        return outputs, logabsdet
    else

        theta = (inputs .- input_cumwidths) ./ input_bin_widths

        numerator = (wa * ya .* (lam .- theta) .+ wc .* yc .* theta) .* float(theta .<= lam) .+ (wc .* yc .* (1 .- theta) .+ wb .* yb .* (theta .- lam)) .* float(theta .> lam)

        denominator = (wa .* (lam .- theta) .+ wc .* theta) .* float(theta .<= lam) .+ (wc .* (1 .- theta) .+ wb .* (theta .- lam)) .* float(theta .> lam)

        outputs = numerator ./ denominator

        derivative_numerator = (wa .* wc .* lam .* (yc .- ya) .* float(theta .<= lam) .+ wb .* wc .* (1 .- lam) .* (yb .- yc) .* float(theta .> lam)) ./ input_bin_widths

        logabsdet = log.(derivative_numerator) .- 2 * log.(abs.(denominator))
    end 

    return outputs, logabsdet
end 