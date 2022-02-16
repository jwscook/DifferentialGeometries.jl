@testset "Cylindrical" begin
  f = ToCartesian(r->[r[1] * cos(r[2]), r[1] * sin(r[2]), r[3]], 3)
  f⁻¹ = FromCartesian(x->[sqrt(x[1]^2 + x[2]^2), atan(x[2], x[1]), x[3]], 3)
  numiters = 1

  @testset "inversions" begin
    r = [π, exp(1), √2] # r, θ, z
    @test f⁻¹(f(r)) ≈ r
    @test f(f⁻¹(r)) ≈ r

    @test invert(f)(r) ≈ f⁻¹(r)
    @test invert(f⁻¹)(r) ≈ f(r)

    v1 = ContravariantVector(z->[prod(z), prod(z)^2, prod(z)^3], f, true)
    v2 = ContravariantVector(z->[prod(z), prod(z)^2, prod(z)^3], f⁻¹, true)
    @test v1(r) ≈ v2(r)
    v1 = CovariantVector(z->[prod(z), prod(z)^2, prod(z)^3], f, true)
    v2 = CovariantVector(z->[prod(z), prod(z)^2, prod(z)^3], f⁻¹, true)
    @test v1(r) ≈ v2(r)
  end

  @testset "metrics, jacobians, line elements" begin
    r = [π, exp(1), √2] # r, θ, z
    @test gᵢⱼ(f, r) ≈ inv(gⁱʲ(f, r))
    @test gᵢⱼ(f⁻¹, r) ≈ inv(gⁱʲ(f⁻¹, r))
    @test gᵢⱼ(f⁻¹, r) ≈ gᵢⱼ(f, r)
    @test gⁱʲ(f⁻¹, r) ≈ gⁱʲ(f, r)
    @test J(f⁻¹, r) ≈ J(f, r)
    @test J⃗(f⁻¹, r) ≈ J⃗(f, r)
    @test dl(f⁻¹, r, Covariant) ≈ dl(f, r, Covariant)
    @test dl(f⁻¹, r, Contravariant) ≈ dl(f, r, Contravariant)
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
    else
      X = randn(3)
      R = f⁻¹(X)
    end
    fR(x) = typeof(ct) <: ToCartesian ? x : f⁻¹(x)
    # assert logic is correct or the tests are meaningless
    @assert f(R) ≈ X
    @assert f⁻¹(X) ≈ R
    @assert fR(X) ≈ R
    return (R,X,fR)
  end

  @testset "metrics, jacobians, line elements" begin
    for ct ∈ (f, f⁻¹), _ in 1:numiters
      (R,X,fR) = getcoordsandtransforms(ct)

      @test gᵢⱼ(ct, R) ≈ [1 0 0; 0 R[1]^2 0; 0 0 1]
      @test gⁱʲ(ct, R) ≈ [1 0 0; 0 1/R[1]^2 0; 0 0 1]
      @test dl(ct, R, Covariant) ≈ [1, 1/R[1], 1]
      @test dl(ct, R, Contravariant) ≈ [1, R[1], 1]

      @test J(ct, R) ≈ R[1]
    end
  end

  @testset "divergence" begin
    for ct ∈ (f, f⁻¹), _ in 1:numiters
      (R,X,fR) = getcoordsandtransforms(ct)

      div10r(r) = [2 * r[1]/2, 3 * r[1] * r[2], 5 * r[3]]
      v = ContravariantVector(z->div10r(fR(z)), ct, true)
      @test div(v)(R) ≈ 10
      if !(div(v)(R) ≈ 10)
        @show ct
      end

      p, m, a = rand(2:5), rand(2:5), rand(2:5)
      fr = r -> r[1]^p
      fθ = r -> sin(m * r[2])
      fz = r -> r[3]^a
      dfr(r) = p * r[1]^(p - 1)
      dfθ(r) = m * cos(m * r[2])
      dfz(r) = a * r[3]^(a - 1)
      divf(r) = dfr(r) + fr(r)/r[1] + dfθ(r)/r[1] + dfz(r)
      v = CovariantVector(r->[fr(r), fθ(r), fz(r)], f, true)
      @test div(v)(R) ≈ divf(R)
      v = ContravariantVector(r->[fr(r), fθ(r), fz(r)], f, true)
      @test div(v)(R) ≈ divf(R)
    end
  end

  @testset "curl" begin
    for ct ∈ (f, f⁻¹), _ in 1:numiters
      (R,X,fR) = getcoordsandtransforms(ct)

      v = CovariantVector(z->[z[3] - 2z[2] * z[1], 0, 0], ct, true)
      @test curl(v)(R, true) ≈ [0, 1, 2]

      a, b = rand(3), rand(2:5, 3, 3)
      fv(z) = [a[i] .* prod(z .^ b[:, i]) for i in 1:3]
      r = [2, 2pi, 2] .* rand(3) .- [0, pi, 1]
      j = ForwardDiff.jacobian(fv, r)
      jr = ForwardDiff.jacobian(x->fv(x) * x[1], r)
      for v ∈ (CovariantVector(fv, ct, true), ContravariantVector(fv, ct, true))
        @test curl(v)(r, true) ≈ [j[3, 2]/r[1] - j[2, 3],
                                  j[1, 3] - j[3, 1],
                                  (jr[2, 1] - j[1, 2]) / r[1]]
      end
    end
  end

  @testset "div curl, curl grad" begin
    for ct ∈ (f, f⁻¹), _ in 1:numiters
      (R,X,fR) = getcoordsandtransforms(ct)
      v = ContravariantVector(z->[prod(z), prod(z)^2, prod(z)^3], ct, true)
      @test div(curl(v))(R) ≈ 0 atol=1e-12
      @test ContravariantVector(CovariantVector(v))(R) ≈ v(R)
      @test CovariantVector(v)(R, true) ≈ v(R, true)
      v = CovariantVector(z->[prod(z), prod(z)^2, prod(z)^3], ct, true)
      @test div(curl(v))(R) ≈ 0 atol=1e-12
      @test CovariantVector(ContravariantVector(v))(R) ≈ v(R)
      @test ContravariantVector(v)(R, true) ≈ v(R, true)

      a, b = rand(3), rand(2:5, 3, 3)
      s(x) = mapreduce(i->a[i] * prod(x.^b[:, i]), +, 1:3)
      @test curl(grad(s, ct))(R) ≈ zeros(3) atol=1e-12
    end
  end

  @testset "endless operations" begin
    (R,X,fR) = getcoordsandtransforms(f)
    a, b = rand(3), rand(4:7, 3, 3)
    fv(z) = [a[i] .* prod(z .^ b[:, i]) for i in 1:3]
    for T1 ∈ (CovariantVector, ContravariantVector)
      for T2 ∈ (CovariantVector, ContravariantVector)
        A = T1(fv, f, true)
        B = T2(fv, f⁻¹, true)
        @test A(R, true) ≈ B(R, true)
        cA = curl(A)(R, true)
        ccA = curl(curl(A))(R, true)
        #cccA = curl(curl(curl(A)))(R, true)
        #ccccA = curl(curl(curl(curl(A))))(R, true)
        cB = curl(B)(R, true)
        ccB = curl(curl(B))(R, true)
        #cccB = curl(curl(curl(B)))(R, true)
        #ccccB = curl(curl(curl(curl(B))))(R, true)
        @test all(x->!iszero(x), cA)
        @test all(x->!iszero(x), ccA)
        #@test all(x->!iszero(x), cccA)
        #@test all(x->!iszero(x), ccccA)
        @test cA    ≈ cB
        @test ccA   ≈ ccB
        #@test cccA  ≈ cccB
        #@test ccccA ≈ ccccB
      end
    end
  end
end


