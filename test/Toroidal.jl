using Test, LinearAlgebra

@testset "Toroidal tests" begin
  numberofiterations = 10

  R0 = 3.0
  Z0 = 0.0
  majorradius(x) = sqrt(sum(x[1:2].^2))
  toroidalangle(x) = atan(x[2], x[1])
  lateral(x) = majorradius(x) - R0
  axial(x) = x[3] - Z0
  minorradius(x) = sqrt(lateral(x)^2 + axial(x)^2)
  poloidalangle(x) = atan(axial(x), lateral(x))
  torus = CoordinateTransform([xyz -> minorradius(xyz),
                               xyz -> toroidalangle(xyz),
                               xyz -> poloidalangle(xyz)])

  @testset "Length of CoordinateTransform" begin
    @test length(torus) == 3
  end

  @testset "Gradient of CoordinateTransform" begin
    for i ∈ 1:numberofiterations
      xyz = rand(3)
      x, y, z = xyz
      r, ϕ, θ = torus(xyz)
      @assert r ≈ sqrt((sqrt(x^2 + y^2) - R0)^2 + (z - Z0)^2)
      drdx = x * (sqrt(x^2 + y^2) - R0) / sqrt(x^2 + y^2) / r
      drdy = y * (sqrt(x^2 + y^2) - R0) / sqrt(x^2 + y^2) / r
      drdz = (z - Z0) / r
      dtdx = - y / (x^2 + y^2)
      dtdy = x / (x^2 + y^2)
      dtdz = 0.0
      dpdx = - x * (z - Z0) / sqrt(x^2 + y^2) / r^2
      dpdy = - y * (z - Z0) / sqrt(x^2 + y^2) / r^2
      dpdz = (sqrt(x^2 + y^2) - R0) / r^2
      @test ∇(torus, 1)(xyz) ≈ [drdx, drdy, drdz]
      @test ∇(torus, 2)(xyz) ≈ [dtdx, dtdy, dtdz]
      @test ∇(torus, 3)(xyz) ≈ [dpdx, dpdy, dpdz]
    end
  end

  @testset "Metric Tensors" begin
    @testset "Contravariant Metric Tensors" begin
      for i ∈ 1:numberofiterations
        xyz = rand(3)
        r, ϕ, θ = torus(xyz)
        R = sqrt(sum(xyz[1:2].^2))
        gij = Array{Float64}(I, 3, 3)
        gij[2, 2] = 1/R^2
        gij[3, 3] = 1/r^2
        @test gⁱʲ(torus)(xyz) ≈ gij
      end
    end
    @testset "Contravariant Metric Tensors" begin
      for i ∈ 1:numberofiterations
        xyz = rand(3)
        r, ϕ, θ = torus(xyz)
        R = sqrt(sum(xyz[1:2].^2))
        g_ij = Array{Float64}(I, 3, 3)
        g_ij[2, 2] = R^2
        g_ij[3, 3] = r^2
        @test gᵢⱼ(torus)(xyz) ≈ g_ij
      end
    end
  end

  @testset "Jacobian" begin
    for i ∈ 1:numberofiterations
      xyz = rand(3)
      r, ϕ, θ = torus(xyz)
      R = sqrt(sum(xyz[1:2].^2))
      @test J(torus)(xyz) ≈ r * R
    end
  end

  @testset "Inverse" begin
    for i ∈ 1:numberofiterations
      r, ϕ, θ = rand(), rand()*2*π - π, rand()*2*π - π
      R = R0 + r * cos(θ)
      x, y, z = R * cos(ϕ), R * sin(ϕ), Z0 + r * sin(θ)
      @test inverse(torus, [r, ϕ, θ]) ≈ [x, y, z] rtol=sqrt(eps())
      @test torus \ [r, ϕ, θ] ≈ [x, y, z] rtol=sqrt(eps())
    end
  end

  @testset "reverse inverse" begin
    function f(rϕθ)
      r, ϕ, θ = rϕθ
      return [(R0 + r * cos(θ)) * cos(ϕ), (R0 + r * cos(θ)) * sin(ϕ), Z0 + r * sin(θ)]
    end
    invtorus = CoordinateTransform(f, 3)
    for i ∈ 1:numberofiterations
      x, y, z = rand(3)*4 .- 2
      @test invtorus(invtorus \ [x, y, z]) ≈ [x, y, z] rtol=sqrt(eps())
    end
  end

  @testset "∂" begin
    for i ∈ 1:numberofiterations
      xyz = rand(3) .* 2 .- 1
      l, n, m = rand(1:5), rand(-5:5), rand(-5:5)
      frϕθ(r, ϕ, θ) = r^l * cos(n*ϕ) * sin(m*θ)
      f(xyz) = frϕθ(torus(xyz)...)
      function dfdr(xyz) r, ϕ, θ = torus(xyz); return l * r^(l-1) * cos(n*ϕ) * sin(m*θ) end
      function dfdϕ(xyz) r, ϕ, θ = torus(xyz); return - r^l * n * sin(n*ϕ) * sin(m*θ) end
      function dfdθ(xyz) r, ϕ, θ = torus(xyz); return r^l * cos(n*ϕ) * m * cos(m*θ) end
      @test ∂(torus, f)(xyz) ≈ [dfdr(xyz), dfdϕ(xyz), dfdθ(xyz)]
    end
  end

end
