const DEFAULT_MIN_BIN_WIDTH = 1e-3
const DEFAULT_MIN_BIN_HEIGHT = 1e-3
const DEFAULT_MIN_DERIVATIVE = 1e-3

function unconstrained_rational_linear_spline(inputs::AbstractArray,
                                         unnormalized_widths::AbstractArray,
                                         unnormalized_heights::AbstractArray,
                                         unnormalized_derivatives::AbstractArray,
                                         unnormalized_lambdas::AbstractArray,
                                         inverse::Bool = false,
                                         tails::String = "linear",
                                         tail_bound::AbstractFloat = 1.,
                                         min_bin_width::AbstractFloat = DEFAULT_MIN_BIN_WIDTH,
                                         min_bin_height::AbstractFloat = DEFAULT_MIN_BIN_HEIGHT,
                                         min_derivative::AbstractFloat = DEFAULT_MIN_DERIVATIVE)

    inside_interval_mask = [(inputs >= -tail_bound) & (inputs <= tail_bound)]
    outside_interval_mask = ~inside_interval_mask

    outputs   = zero(inputs)
    logabsdet = zero(inputs)

    if tails == "linear"    
        pushfirst!(unnormalized_derivatives, 1)
        push!(unnormalized_derivatives, 1)
        constant = log(exp(1 - min_derivative) - 1)
        unnormalized_derivatives[.., 1] = constant
        unnormalized_derivatives[.., length(unnormalized_derivatives)] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else
        Error("tails are not implemented?")
    end 

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_linear_spline(
                                inputs[inside_interval_mask],
                                unnormalized_widths[inside_interval_mask, :],
                                unnormalized_heights[inside_interval_mask, :],
                                unnormalized_derivatives[inside_interval_mask, :],
                                unnormalized_lambdas[inside_interval_mask, :],
                                inverse,
                                left = -tail_bound, 
                                right = tail_bound, 
                                bottom = -tail_bound, 
                                top = tail_bound,
                                min_bin_width,
                                min_bin_height,
                                min_derivative)


    return outputs, logabsdet
end 

function rational_linear_spline(inputs::AbstractArray,
                                unnormalized_widths::AbstractArray,
                                unnormalized_heights::AbstractArray,
                                unnormalized_derivatives::AbstractArray,
                                unnormalized_lambdas::AbstractArray,
                                inverse::Bool = false,
                                left::AbstractFloat = 0., 
                                right::AbstractFloat = 1., 
                                bottom::AbstractFloat = 0., 
                                top::AbstractFloat = 1.,
                                min_bin_width::AbstractFloat = DEFAULT_MIN_BIN_WIDTH,
                                min_bin_height::AbstractFloat = DEFAULT_MIN_BIN_HEIGHT,
                                min_derivative::AbstractFloat = DEFAULT_MIN_DERIVATIVE)

    if minimum(inputs) < left | maximum(inputs) > right
        Error("InputOutsideDomain")
    end 

    num_bins = length(unnormalized_widths)

    if min_bin_width * num_bins > 1.0
        Error("Minimal bin width too large for the number of bins")
    end 

    if min_bin_height * num_bins > 1.0
        Error("Minimal bin height too large for the number of bins")
    end 

    widths = softmax(unnormalized_widths, ndims(unnormalized_widths))
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    cumwidths = cumsum(widths)                 #cumsum(widths, ndims(widths))
    #cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    pushfirst!(cumwidths, 1)
    push!(cumwidths, 0)
    cumwidths = (right - left) * cumwidths + left

    cumwidths[.., 0] = left
    cumwidths[.., length(cumwidths)] = right
    widths = cumwidths[.., 2:end] - cumwidths[.., 1:end]

    derivatives = min_derivative + softplus(unnormalized_derivatives)

    heights = softmax(unnormalized_heights, ndims(unnormalized_heights))
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = cumsum(heights)               #cumsum(heights, ndims(heights))
    #cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    pushfirst!(cumheights, 1)
    push!(cumheights, 0)

    cumheights = (top - bottom) * cumheights + bottom
    cumheights[.., 1] = bottom
    cumheights[.., length(cumheights)] = top
    heights = cumheights[.., 2:end] - cumheights[.., 1:end]

    # watch carefully if this works: (original uses np.newaxis)
    if inverse
        bin_idx = searchsortedfirst.(cumheights, inputs)
    else
        bin_idx = searchsortedfirst.(cumwidths, inputs)
    end 

    input_cumwidths = selectdim(cumwidhts, ndims(cumwidths), bin_idx)[.., 1]
    input_bin_widths = selectdim(widhts, ndims(widths), bin_idx)[.., 1]

    input_cumheights = cselectdim(cumheights, ndims(cumheights), bin_idx)[.., 1]
    delta = heights / widths
    input_delta = selectdim(delta, ndims(delta), bin_idx)[.., 1]

    input_derivatives = selectdim(derivatives, ndims(derivatives), bin_idx)[.., 1]
    input_derivatives_plus_one = selectdim(derivatives[.., 2:end], ndims(derivatives[.., 2:end]), bin_idx)[.., 1]

    input_heights = selectdim(heigths, ndims(heights), bin_idx)[.., 1]

    lambdas = 0.95 * sigmoid(unnormalized_lambdas) + 0.025

    lam = selectdim(lambdas, ndims(lambdas), bin_idx)[.., 1]
    wa  = 1
    wb  = sqrt(input_derivatives/input_derivatives_plus_one) * wa
    wc  = (lam * wa * input_derivatives + (1-lam) * wb * input_derivatives_plus_one)/input_delta
    ya  = input_cumheights
    yb  = input_heights + input_cumheights
    yc  = ((1-lam) * wa * ya + lam * wb * yb)/((1-lam) * wa + lam * wb)

    if inverse

        numerator = (lam * wa * (ya - inputs)) * float(inputs <= yc) +  ((wc - lam * wb) * inputs + lam * wb * yb - wc * yc) * float(inputs > yc)

        denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * float(inputs <= yc) + ((wc - wb) * inputs + wb * yb - wc * yc) * float(inputs > yc)

        theta = numerator/denominator

        outputs = theta * input_bin_widths + input_cumwidths

        derivative_numerator = (wa * wc * lam * (yc - ya) * float(inputs <= yc) + wb * wc * (1 - lam) * (yb - yc) * float(inputs > yc)) * input_bin_widths

        logabsdet = log(derivative_numerator) - 2 * log(abs(denominator))

        return outputs, logabsdet
    else

        theta = (inputs - input_cumwidths) / input_bin_widths

        numerator = (wa * ya * (lam - theta) + wc * yc * theta) * float(theta <= lam) + (wc * yc * (1 - theta) + wb * yb * (theta - lam)) * float(theta > lam)

        denominator = (wa * (lam - theta) + wc * theta) * float(theta <= lam) + (wc * (1 - theta) + wb * (theta - lam)) * float(theta > lam)

        outputs = numerator / denominator

        derivative_numerator = (wa * wc * lam * (yc - ya) * float(theta <= lam) + wb * wc * (1 - lam) * (yb - yc) * float(theta > lam)) / input_bin_widths

        logabsdet = log(derivative_numerator) - 2 * log(abs(denominator))
    end 

    return outputs, logabsdet
end 