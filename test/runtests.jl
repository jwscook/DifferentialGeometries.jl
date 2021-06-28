using DifferentialGeometries
using ForwardDiff, LinearAlgebra, Random, Test

Random.seed!(0)

@testset "DifferentialGeometries tests" begin
  include("Cartesian.jl")
  include("Cylindrical.jl")
  include("Toroidal.jl")
  include("ChangeOfCoordinates.jl")
  include("AutodiffPropagation.jl")
end
