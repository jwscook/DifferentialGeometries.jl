using Test, LinearAlgebra

@testset "Polar2D tests" begin
  numberofiterations = 10

  polar = CoordinateTransform([xy -> sqrt(xy[1].^2 + xy[2].^2),
                               xy -> atan(xy[2], xy[1])])

  @testset "Length of CoordinateTransform" begin
    @test length(polar) == 2
  end

  @testset "Gradient of CoordinateTransform" begin
    for i ∈ 1:numberofiterations
      xy = rand(2)
      x, y = xy
      r, θ = polar(xy)
      @test ∇(polar, 1)(xy) ≈ xy ./ r
      @test ∇(polar, 2)(xy) ≈ [-y ./ r^2, x / r^2]
    end
  end

  @testset "Metric Tensors" begin
    @testset "Contravariant Metric Tensors" begin
      for i ∈ 1:numberofiterations
        xy = rand(2)
        r, θ = polar(xy)
        gij = Array{Float64}(I, 2, 2)
        gij[2, 2] = 1/r^2
        @test gⁱʲ(polar)(xy) ≈ gij
      end
    end
    @testset "Contravariant Metric Tensors" begin
      for i ∈ 1:numberofiterations
        xy = rand(2)
        r, θ = polar(xy)
        g_ij = Array{Float64}(I, 2, 2)
        g_ij[2, 2] = r^2
        @test gᵢⱼ(polar)(xy) ≈ g_ij
      end
    end
  end

  @testset "Jacobian" begin
    for i ∈ 1:numberofiterations
      xy = rand(2)
      r, θ = polar(xy)
      @test J(polar)(xy) ≈ r
    end
  end

  @testset "inverse" begin
    for i ∈ 1:numberofiterations
      r, θ = rand(), rand()*2*π - π
      x, y = r*cos(θ), r*sin(θ)
      @test inverse(polar, [r, θ]) ≈ [x,  y] rtol=sqrt(eps())
      @test polar \ [r, θ] ≈ [x,  y] rtol=sqrt(eps())
    end
  end

  @testset "reverse inverse" begin
    invpolar = CoordinateTransform([rθ -> rθ[1]*cos(rθ[2]),
                                    rθ -> rθ[1]*sin(rθ[2])])
    for i ∈ 1:numberofiterations
      r, θ = rand(), rand()*2*π - π
      x, y = r*cos(θ), r*sin(θ)
      rθ = invpolar \ [x, y]
      @test [abs(rθ[1]), mod(rθ[2], π)] ≈ [r, mod(θ, π)] rtol=sqrt(eps())
      @test invpolar(invpolar \ [x, y]) ≈ [x, y] rtol=sqrt(eps())
    end
  end

  @testset "∂" begin
    for i ∈ 1:numberofiterations
      xy = rand(2) .* 2 .- 1
      n, m = rand(1:5), rand(-5:5)
      frθ(r, θ) = r^n * sin(m*θ)
      f(xy) = frθ(polar(xy)...)
      function dfdr(xy) r, θ = polar(xy); return n * r^(n-1) * sin(m*θ) end
      function dfdθ(xy) r, θ = polar(xy); return r^n * m * cos(m*θ) end
      @test ∂(polar, f)(xy) ≈ [dfdr(xy), dfdθ(xy)]
    end
  end

  @testset "div" begin
    for i ∈ 1:numberofiterations
      p, m = rand(0:5), rand(-5:5)
      fr = xy -> polar(xy)[1]^p
      fθ = xy -> sin(m * polar(xy)[2])
      fs = [fr, fθ]
      divfr(xy) = (p + 1) * polar(xy)[1]^(p - 1)
      divfθ(xy) = m * cos(m * polar(xy)[2])
      divf(x) = divfr(x) + divfθ(x)
      xy = rand(2) * 2 .- 1
      @test div(polar, fs)(xy) ≈ divf(xy)
    end
  end

  @testset "curl" begin
    for i ∈ 1:numberofiterations
      p, m = rand(0:5), rand(-5:5)
      fr = xy -> polar(xy)[1]^p
      fθ = xy -> sin(m * polar(xy)[2])
      fs = [fr, fθ]
      divfr(xy) = (p + 1) * polar(xy)[1]^(p - 1)
      divfθ(xy) = m * cos(m * polar(xy)[2])
      divf(x) = divfr(x) + divfθ(x)
      xy = rand(2) * 2 .- 1
      @test div(polar, fs)(xy) ≈ divf(xy)
    end
  end


end
