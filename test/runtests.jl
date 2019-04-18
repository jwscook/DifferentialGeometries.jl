#include("../src/implementation.jl")
using DifferentialGeometries
using Test, LinearAlgebra, ForwardDiff

@testset "Cartesian" begin
  x = [π, exp(1), √2]
  for ct ∈ (ToCartesian(x->x, 3), FromCartesian(x->x, 3))
    @test J(ct, x) == 1
    m = zeros(3,3) + I
    gᵢⱼ(ct, x) ≈ m
    gⁱʲ(ct, x) ≈ m
    dl(ct, x, Covariant) ≈ ones(3)
    dl(ct, x, Contravariant) ≈ ones(3)

    v = CovariantVector(x->[x[1], x[2], x[3]], ct, true)
    @test div(v)(x) ≈ 3
    @test curl(v)(x) == zeros(3)
    v = CovariantVector(x->[x[3], x[1], x[2]], ct, true)
    @test div(v)(x) == 0
    @test curl(v)(x) == ones(3)
    @test div(curl(v))(x) == 0
    @test div(curl(v))(x) == 0
    v = CovariantVector(x->[-x[2], -x[3], -x[1]], ct, true)
    @test div(v)(x) == 0
    @test curl(v)(x) == ones(3)
    @test div(curl(v))(x) == 0
    @test div(curl(v))(x) == 0
  end
end


@testset "Cylindrical" begin
  f = ToCartesian(r->[r[1] * cos(r[2]), r[1]*sin(r[2]), r[3]], 3)
  f⁻¹ = FromCartesian(x->[sqrt(x[1]^2 + x[2]^2), atan(x[2], x[1]), x[3]], 3)

  @testset "inversions" begin
    r = [π, exp(1), √2] # r, θ, z
    @test f⁻¹(f(r)) ≈ r
    @test f(f⁻¹(r)) ≈ r

    @test invert(f)(r) ≈ f⁻¹(r)
    @test invert(f⁻¹)(r) ≈ f(r)
  end

  @testset "metrics, jacobians, line elements" begin
    r = [π, exp(1), √2] # r, θ, z
    x = f(r) # x, y, z
    @test gᵢⱼ(f, r) ≈ inv(gⁱʲ(f, r))
    @test gᵢⱼ(f⁻¹, x) ≈ inv(gⁱʲ(f⁻¹, x))
    @test gᵢⱼ(f⁻¹, x) ≈ gᵢⱼ(f, r)
    @test gⁱʲ(f⁻¹, x) ≈ gⁱʲ(f, r)
    @test J(f⁻¹, x) ≈ J(f, r)
    @test J⃗(f⁻¹, x) ≈ J⃗(f, r)
    @test dl(f⁻¹, x, Covariant) ≈ dl(f, r, Covariant)
    @test dl(f⁻¹, x, Contravariant) ≈ dl(f, r, Contravariant)
  end

  @testset "unit basis / physical units expressions" begin
    r = [π, exp(1), √2] # r, θ, z
    x = f(r) # x, y, z
    g(z) = [prod(z), prod(z)^2, prod(z)^3]
    v1 = ContravariantVector(z->g(z), f, true)
    v2 = ContravariantVector(z->g(f⁻¹(z)), f⁻¹, true)
    @test v1(r, true) ≈ v2(x, true)

    v1 = CovariantVector(z->g(z), f, true)
    v2 = CovariantVector(z->g(f⁻¹(z)), f⁻¹, true)
    @test v1(r, true) ≈ v2(x, true)
  end


  function getcoordsandtransforms(ct)
    if typeof(ct) <: ToCartesian
      R = [rand(), (rand() - 0.5) * 2π , randn()]
      X = f(R)
      ξ = R
    else
      X = randn(3)
      R = f⁻¹(X)
      ξ = X
    end
    fR(x) = typeof(ct) <: ToCartesian ? x : f⁻¹(x)
    # assert logic is correct or the tests are meaningless
    @assert f(R) ≈ X
    @assert f⁻¹(X) ≈ R
    (typeof(ct) <: FromCartesian) && @assert ξ ≈ X
    (typeof(ct) <: ToCartesian) && @assert ξ ≈ R
    return (R,X,ξ,fR)
  end

  @testset "metrics, jacobians, line elements" begin
    for ct ∈ (f, f⁻¹), _ in 1:10
      (R,X,ξ,fR) = getcoordsandtransforms(ct)

      @test gᵢⱼ(ct, ξ) ≈ [1 0 0; 0 R[1]^2 0; 0 0 1]
      @test gⁱʲ(ct, ξ) ≈ [1 0 0; 0 1/R[1]^2 0; 0 0 1]
      @test dl(ct, ξ, Covariant) ≈ [1, 1/R[1], 1]
      @test dl(ct, ξ, Contravariant) ≈ [1, R[1], 1]

      @test J(ct, ξ) ≈ R[1]
    end
  end

  @testset "divergence" begin
    #for ct ∈ (f, f⁻¹), _ in 1:10
    for ct ∈ (f, ), _ in 1:10
      (R,X,ξ,fR) = getcoordsandtransforms(ct)

      div10r(r) = [2 * r[1]/2, 3 * r[1] * r[2], 5 * r[3]]
      v = ContravariantVector(z->div10r(fR(z)), ct, true)
      @test div(v)(ξ) ≈ 10
      @test div(v)(R) ≈ 10
      #@test div(v)(X) ≈ 10 # should be 10 for everything, except -ve radii

      p, m, a = rand(2:5), rand(2:5), rand(2:5)
      fr = r -> r[1]^p
      fθ = r -> sin(m * r[2])
      fz = r -> r[3]^a
      dfr(r) = p * r[1]^(p - 1)
      dfθ(r) = m * cos(m * r[2])
      dfz(r) = a * r[3]^(a - 1)
      divf(r) = dfr(r) + fr(r)/r[1] + dfθ(r)/r[1] + dfz(r)
      v = CovariantVector(x->(r=fR(x); [fr(r), fθ(r), fz(r)]), f, true)
      @test div(v)(ξ) ≈ divf(fR(ξ))
      v = ContravariantVector(x->(r=fR(x); [fr(r), fθ(r), fz(r)]), f, true)
      @test div(v)(ξ) ≈ divf(fR(ξ))
    end
  end

  @testset "curl" begin
    for ct ∈ (f, f⁻¹), _ in 1:10
      (R,X,ξ,fR) = getcoordsandtransforms(ct)

      v = CovariantVector(z->(z=fR(z); [z[3] - 2z[2] * z[1], 0, 0]), ct, true)
      @test curl(v)(ξ, true) ≈ [0, 1, 2]

      a, b = rand(3), rand(2:5, 3, 3)
      fv = z -> [a .* prod(z .^ b[:, i]) for i in 1:3]
      r = [2, 2pi, 2] .* rand(3) .- [0, pi, 1]
      x = f(r)
      j = ForwardDiff.jacobian(fv, r)
      jr = ForwardDiff.jacobian(x->fv(x) * x[1], r)
      v = CovariantVector(fv, f, true)
      @test curl(v)(r, true) ≈ [j[3, 2]/r[1] - j[2, 3],
                                j[1, 3] - j[3, 1],
                                (jr[2, 1] - j[1, 2]) / r[1]]
    end
  end

  @testset "div curl, curl grad" begin
    for ct ∈ (f, f⁻¹), _ in 1:10
      (R,X,ξ,fR) = getcoordsandtransforms(ct)
      v = ContravariantVector(z->[prod(z), prod(z)^2, prod(z)^3], ct, true)
      @test div(curl(v))(ξ) ≈ 0 atol=1e-12
      @test ContravariantVector(CovariantVector(v))(ξ) ≈ v(ξ)
      @test CovariantVector(v)(ξ, true) ≈ v(ξ, true)
      v = CovariantVector(z->[prod(z), prod(z)^2, prod(z)^3], ct, true)
      @test div(curl(v))(ξ) ≈ 0 atol=1e-12
      @test CovariantVector(ContravariantVector(v))(ξ) ≈ v(ξ)
      @test ContravariantVector(v)(ξ, true) ≈ v(ξ, true)

      a, b = rand(3), rand(2:5, 3, 3)
      s(x) = mapreduce(i->a[i] * prod(x.^b[:, i]), +, 1:3)
      @test curl(grad(s, ct))(ξ) ≈ zeros(3) atol=1e-12
    end
  end
end


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
      @test J⃗(f⁻¹, xyz, Covariant) ≈ [drdx drdy drdz;
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
        @test gᵢⱼ(f⁻¹, xyz) ≈ g_ij
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
      @test J(f⁻¹, xyz) ≈ r * R
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
