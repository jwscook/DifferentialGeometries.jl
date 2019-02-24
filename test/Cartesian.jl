using Test, LinearAlgebra

@testset "Cartesian tests" begin
  numberofiterations = 1

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

  @testset "div" begin
    for i ∈ 1:numberofiterations
      a, b, c = rand(0:5), rand(0:5), rand(0:5)
      fa = x->x[1]^a + 2 + exp(-x[2])
      fb = x->x[2]^b - 3 + x[3]^rand()
      fc = x->x[3]^c + 65 + sin(x[1])
      fs = [fa, fb, fc]
      x = rand(3)
      divfa(x) = a * x[1]^(a-1)
      divfb(x) = b * x[2]^(b-1)
      divfc(x) = c * x[3]^(c-1)
      @test ∂(xyz, fa)(x)[1] ≈ divfa(x)
      @test ∂(xyz, fb)(x)[2] ≈ divfb(x)
      @test ∂(xyz, fc)(x)[3] ≈ divfc(x)
      divf(x) = divfa(x) + divfb(x) + divfc(x)
      @test length(div(xyz, fs)(x)) == 1
      @test div(xyz, fs)(x) ≈ divf(x)
    end
  end

  @testset "curl" begin
    for i ∈ 1:numberofiterations
      a1, a2, a3 = rand(0:5), rand(0:5), rand(0:5)
      b1, b2, b3 = rand(0:5), rand(0:5), rand(0:5)
      c1, c2, c3 = rand(0:5), rand(0:5), rand(0:5)
      fa = x -> x[1]^a1 + x[2]^a2 + x[3]^a3
      fb = x -> x[1]^b1 + x[2]^c2 + x[3]^b3
      fc = x -> x[1]^c1 + x[2]^b2 + x[3]^c3
      fs = [fa, fb, fc]
      grad(f) = x -> ForwardDiff.gradient(f, x)
      curlfa(x) = grad(fc)(x)[2] - grad(fb)(x)[3]
      curlfb(x) = grad(fa)(x)[3] - grad(fc)(x)[1]
      curlfc(x) = grad(fb)(x)[1] - grad(fa)(x)[2]
      curlf(x) = [curlfa(x), curlfb(x), curlfc(x)]
      x = rand(3)*2 .- 1

      Aᵢ = Cov(xyz, fs)
      @show ∇(Aᵢ.ct)(x)
      @show inv(∇(Aᵢ.ct)(x))
      @show Aᵢ[1](x)
      @show (∇(y->Aᵢ[1](y)))(x)
      @show ∂(Aᵢ, 1)(x)
      @show ∂(Aᵢ, 2)(x)
      @show ∂(Aᵢ, 3)(x)
      @show ∇(y->Aᵢ(y))(x)
      @show ∇(y->Aᵢ(y))(x)
      @show curlf(x)
      @show curl(xyz, fs)(x)
      @test curl(xyz, fs)(x)' ≈ curlf(x)
    end
  end
#
#  @testset "normalise" begin
#    for i ∈ 1:numberofiterations
#      con = Con(xyz, [x->x[1], x->x[2], x->x[3]])
#      x = rand(3)*2 .- 1
#      @show con(x)
#      ncon = norm(con)
#      @show ncon(x)
#      @test ncon(x) ≈ con(x)
#    end
#  end
#
  @testset "dot" begin
    for i ∈ 1:numberofiterations
    end
  end

end
