# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).


struct WithLADJ end

function (f::Base.ComposedFunction)(x, ::WithLADJ)
    y_inner, ladj_inner = f.inner(x, WithLADJ())
    y, ladj_outer = f.outer(y_inner, WithLADJ())
    (y, ladj_inner + ladj_outer)
end


const ZERO = false
