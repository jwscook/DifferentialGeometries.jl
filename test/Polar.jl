using Test, LinearAlgebra

rθ = CoordinateTransform([xy -> sqrt(xy[1].^2 + xy[2].^2), xy -> atan(xy[2], xy[1])])
@testset "Length of CoordinateTransform" begin
  @test length(rθ) == 2
end
@testset "Gradient of CoordinateTransform" begin
  for i ∈ 1:10
    xy = rand(2)
    x, y = xy
    r, θ = rθ(xy)
    @test ∇(rθ, 1)(xy) ≈ xy ./ r
    @test ∇(rθ, 2)(xy) ≈ [-y ./ r^2, x / r^2]
  end
end

@testset "Metric Tensors" begin
  @testset "Contravariant Metric Tensors" begin
    for i ∈ 1:10
      xy = rand(2)
      r, θ = rθ(xy)
      gij = Array{Float64}(I, 2, 2)
      gij[2, 2] = 1/r^2
      @test gⁱʲ(rθ)(xy) ≈ gij
    end
  end
  @testset "Contravariant Metric Tensors" begin
    for i ∈ 1:10
      xy = rand(2)
      r, θ = rθ(xy)
      g_ij = Array{Float64}(I, 2, 2)
      g_ij[2, 2] = r^2
      @test gᵢⱼ(rθ)(xy) ≈ g_ij
    end
  end
end
 
@testset "Jacobian" begin
  for i ∈ 1:10
    xy = rand(2)
    r, θ = rθ(xy)
    @test J(rθ)(xy) ≈ r
  end
end

@testset "cartesian" begin
  for i ∈ 1:1
    co = [2*rand(2), 2π*rand() - π]
    x, y = co[1]*sin(co[2]), co[1]*cos(co[2])
    @test cartesian(rθ, co) ≈ [x, y]
  end
end

