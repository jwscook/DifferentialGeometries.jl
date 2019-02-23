using Random, Test
@testset "DifferentialGeometry Tests" begin
Random.seed!(18)
include("../src/implementation.jl")
include("TypeSafety.jl")
include("Cartesian.jl")
#include("Polar2D.jl")
#include("Polar3D.jl")
#include("Toroidal.jl")
end
