using Test, LinearAlgebra

@testset "Polar3D tests" begin
  numberofiterations = 10

  polar = CoordinateTransform([xyz -> sqrt(xyz[1].^2 + xyz[2].^2),
                               xyz -> atan(xyz[2], xyz[1]),
                               xyz -> xyz[3]])

  @testset "Length of CoordinateTransform" begin
    @test length(polar) == 3
  end

  @testset "div" begin
    for i ∈ 1:numberofiterations
      p, m, a = rand(0:5), rand(-5:5), rand(0:5)
      fr = xyz -> polar(xyz)[1]^p
      fθ = xyz -> sin(m * polar(xyz)[2])
      fz = xyz -> polar(xyz)[3]^a
      fs = [fr, fθ, fz]
      divfr(xyz) = (p + 1) * polar(xyz)[1]^(p - 1)
      divfθ(xyz) = m * cos(m * polar(xyz)[2])
      divfz(xyz) = a * polar(xyz)[3]^(a - 1)
      divf(x) = divfr(x) + divfθ(x) + divfz(x)
      xyz = rand(3) * 2 .- 1
      @test div(polar, fs)(xyz) ≈ divf(xyz)
    end
  end

  #@testset "curl" begin
  #  for i ∈ 1:numberofiterations
  #    a = [rand(0:5), rand(0:5), rand(0:5)]
  #    b = [rand(0:5), rand(0:5), rand(0:5)]
  #    c = [rand(0:5), rand(0:5), rand(0:5)]
  #    fr = xyz -> sum(polar(xyz).^a)
  #    fθ = xyz -> sum(polar(xyz).^b)
  #    fz = xyz -> sum(polar(xyz).^c)
  #    fs = [fr, fθ, fz]
  #    function curlf(xyz)
  #      r, θ, z = polar(xyz)
  #      curlfr = c[2]*θ^(c[2] - 1) / r - b[3]*z^(b[3]-1)
  #      curlfθ = a[3]*z^(a[3] - 1) - c[1]*r^(c[1]-1)
  #      curlfz = (b[1]*θ^b[1] - a[1]*θ^(a[1]-1)) / r
  #      return [curlfr, curlfθ, curlfz]
  #    end
  #    xyz = rand(3) * 2 .- 1
  #    @test curl(polar, fs)(xyz) ≈ curlf(xyz)
  #  end
  #end


end
