using Test, LinearAlgebra

@testset "Cartesian tests" begin
xyz = CoordinateTransform([xyz->xyz[1], xyz->xyz[2], xyz->xyz[3]])
@testset "Length of CoordinateTransform" begin
  @test length(xyz) == 3
end
@testset "Gradient of CoordinateTransform" begin
  for i ∈ 1:10
    @test ∇(xyz, 1)(rand(3)) == [1.0, 0.0, 0.0]
    @test ∇(xyz, 2)(rand(3)) == [0.0, 1.0, 0.0]
    @test ∇(xyz, 3)(rand(3)) == [0.0, 0.0, 1.0]
  end
end

@testset "Metric Tensors" begin
  @testset "Contravariant Metric Tensors" begin
    for i ∈ 1:10
      @test gⁱʲ(xyz)(rand(3)) ≈ Array{Float64}(I, 3, 3)
    end
  end
  @testset "Contravariant Metric Tensors" begin
    for i ∈ 1:10
      @test gᵢⱼ(xyz)(rand(3)) ≈ Array{Float64}(I, 3, 3)
    end
  end
end
 
@testset "Jacobian" begin
  for i ∈ 1:10
    @test J(xyz)(rand(3)) ≈ 1.0
  end
end
 
@testset "inverse" begin
  for i ∈ 1:10
    x = rand(3)
    @test inverse(xyz, x) ≈ x rtol=sqrt(eps())
  end
end
end
