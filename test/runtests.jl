# This file is a part of EuclidianNormalizingFlows.jl, licensed under the MIT License (MIT).

import Test

Test.@testset "Package EuclidianNormalizingFlows" begin
    include("test_householder_trafo.jl")
    include("test_center_stretch.jl")
    include("test_johnson_trafo.jl")
end # testset
