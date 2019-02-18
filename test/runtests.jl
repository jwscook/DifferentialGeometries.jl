using Random, Test
@testset "DifferentialGeometry Tests" begin
Random.seed!(18)
include("../src/implementation.jl")
include("Cartesian.jl")
include("Polar.jl")
include("Toroidal.jl")
end
