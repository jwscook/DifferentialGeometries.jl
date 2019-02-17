using Test, LinearAlgebra

@testset "Cartesian tests" begin
  numberofiterations = 10

  xyz = CoordinateTransform([xyz->xyz[1], xyz->xyz[2], xyz->xyz[3]])

  @testset "Length of CoordinateTransform" begin
    @test length(xyz) == 3
  end

  @testset "Gradient of CoordinateTransform" begin
    for i ∈ 1:numberofiterations
      @test ∇(xyz, 1)(rand(3)) == [1.0, 0.0, 0.0]
      @test ∇(xyz, 2)(rand(3)) == [0.0, 1.0, 0.0]
      @test ∇(xyz, 3)(rand(3)) == [0.0, 0.0, 1.0]
    end
  end

  @testset "Metric Tensors" begin
    @testset "Contravariant Metric Tensors" begin
      for i ∈ 1:numberofiterations
        @test gⁱʲ(xyz)(rand(3)) ≈ Array{Float64}(I, 3, 3)
      end
    end
    @testset "Contravariant Metric Tensors" begin
      for i ∈ 1:numberofiterations
        @test gᵢⱼ(xyz)(rand(3)) ≈ Array{Float64}(I, 3, 3)
      end
    end
  end

  @testset "Jacobian" begin
    for i ∈ 1:numberofiterations
      @test J(xyz)(rand(3)) ≈ 1.0
    end
  end

  @testset "inverse" begin
    for i ∈ 1:numberofiterations
      x = rand(3)
      @test inverse(xyz, x) ≈ x rtol=sqrt(eps())
      @test xyz \ x ≈ x rtol=sqrt(eps())
    end
  end

  @testset "∂" begin
    for i ∈ 1:numberofiterations
      f(x) = x[1] + x[2]^2 + x[3]^3
      x = rand(3)
      @test ∂(xyz, f)(x) ≈ [1.0, 2.0*x[2], 3.0*x[3]^2]
    end
  end

end
