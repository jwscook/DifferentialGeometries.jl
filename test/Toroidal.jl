@testset "Toroidal tests" begin

  R0 = rand()
  Z0 = rand() - 0.5
  majorradius(x) = sqrt(sum(x[1:2].^2))
  toroidalangle(x) = atan(x[2], x[1])
  lateral(x) = majorradius(x) - R0
  axial(x) = x[3] - Z0
  minorradius(x) = sqrt(lateral(x)^2 + axial(x)^2)
  poloidalangle(x) = atan(axial(x), lateral(x))
  f⁻¹ = FromCartesian(xyz -> [minorradius(xyz),
                              toroidalangle(xyz),
                              poloidalangle(xyz)], 3)

  function r2x(rϕθ)
    r, ϕ, θ = rϕθ
    R = R0 + r * cos(θ)
    x, y, z = R * cos(ϕ), R * sin(ϕ), Z0 + r * sin(θ)
    return [x, y, z]
  end
  f = ToCartesian(r2x, 3)

  @testset "Length of CoordinateTransform" begin
    @test length(f⁻¹) == 3
  end

  @testset "Gradient of CoordinateTransform" begin
    for i ∈ 1:10
      xyz = rand(3) .* 2 .- 1
      x, y, z = xyz
      r, ϕ, θ = f⁻¹(xyz)
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
      @test J⃗(f⁻¹, [r, ϕ, θ], Covariant) ≈ [drdx drdy drdz;
                                      dtdx dtdy dtdz;
                                      dpdx dpdy dpdz]
    end
  end

  @testset "Metric Tensors" begin
    @testset "Contravariant Metric Tensors" begin
      for i ∈ 1:10
        xyz = rand(3) .* 2 .- 1
        x, y, z = xyz
        rϕθ = f⁻¹(xyz)
        r, ϕ, θ = rϕθ
        R = sqrt(sum(xyz[1:2].^2))
        gij = Array{Float64}(I, 3, 3)
        gij[2, 2] = 1/R^2
        gij[3, 3] = 1/r^2
        @test gⁱʲ(f, rϕθ) ≈ gij
      end
    end
    @testset "Contravariant Metric Tensors" begin
      for i ∈ 1:10
        xyz = rand(3) .* 2 .- 1
        x, y, z = xyz
        r, ϕ, θ = f⁻¹(xyz)
        R = sqrt(sum(xyz[1:2].^2))
        g_ij = Array{Float64}(I, 3, 3)
        g_ij[2, 2] = R^2
        g_ij[3, 3] = r^2
        @test gᵢⱼ(f⁻¹, [r, ϕ, θ]) ≈ g_ij
      end
    end
  end

  @testset "Jacobian" begin
    for i ∈ 1:10
      xyz = rand(3) .* 2 .- 1
      x, y, z = xyz
      rϕθ = f⁻¹(xyz)
      r, ϕ, θ = rϕθ
      R = R0 + r * cos(θ)
      @test J(f, rϕθ) ≈ r * R
      @test J(f⁻¹, rϕθ) ≈ r * R
    end
  end

  @testset "invert" begin
    for i ∈ 1:10
      xyz = rand(3) .* 2 .- 1
      rϕθ = f⁻¹(xyz)
      @test f⁻¹(f(rϕθ)) ≈ rϕθ
      @test f(f⁻¹(xyz)) ≈ xyz
      @test invert(f⁻¹, rϕθ) ≈ xyz
      @test f⁻¹ \ rϕθ ≈ xyz
    end
  end

  @testset "divergence" begin
      a, b = rand(3), rand(2:5, 3, 3)
      fv = z -> [a .* prod(z .^ b[:, i]) for i in 1:3]
      r = [2, 2pi, 2] .* rand(3) .- [0, pi, 1]
      x = f(r)
      j = ForwardDiff.jacobian(fv, r)
      jr = ForwardDiff.jacobian(x->fv(x) * x[1], r)
      R = sqrt(x[1]^2 + x[2]^2)
      divv = jr[1, 1] / r[1] + j[2,2] / R + j[3,3] / r[1]

      v = CovariantVector(fv, f, true)
      @test div(v)(r) ≈ divv
      v = ContravariantVector(fv, f, true)
      @test div(v)(r) ≈ divv
  end

  @testset "curl" begin
      a, b = rand(3), rand(2:5, 3, 3)
      fv = z -> [a .* prod(z .^ b[:, i]) for i in 1:3]
      r = [2, 2pi, 2] .* rand(3) .- [0, pi, 1]
      x = f(r)
      j = ForwardDiff.jacobian(fv, r)
      jr = ForwardDiff.jacobian(x->fv(x) * x[1], r)
      R = sqrt(x[1]^2 + x[2]^2)
      curl1 = (j[3, 2] - j[2, 3]) / r[1]
      curl2 = (j[1, 3] - jr[3, 1]) / r[1]
      curl3 = j[2, 1] - j[1, 2] / R
      curlv = [curl1, curl2, curl3]

      v = CovariantVector(fv, f, true)
      @test curl(v)(r, true) ≈ curlv
      v = ContravariantVector(fv, f, true)
      @test curl(v)(r, true) ≈ curlv
  end
end
