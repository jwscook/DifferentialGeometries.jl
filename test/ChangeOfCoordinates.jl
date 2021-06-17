using DifferentialGeometries: Cov, Con

@testset "Changing coordinates: 2D polar to 2D hyperbolic" begin
  c = ToCartesian(r->[r[1] * cos(r[2]), r[1] * sin(r[2])], 2)
  c⁻¹ = FromCartesian(x->[sqrt(x[1]^2 + x[2]^2), atan(x[2], x[1])], 2)

  h = ToCartesian(r->[r[2] * exp(r[1]), r[2] * exp(-r[1])], 2)
  h⁻¹ = FromCartesian(x->[log(sqrt(x[1] / x[2])), sqrt(x[1] * x[2])], 2)

  x = rand(2)
  r = c⁻¹(x)
  u = h \ x
  # check some logic else it's all for nought
  @assert h(u) ≈ x
  @assert h⁻¹(x) ≈ u
  @assert c⁻¹(h(u)) ≈ r
  @assert c \ h(u) ≈ r
  @assert h \ (c(r)) ≈ u

  f(Rθ) = ((R,θ)=Rθ; return [R^2 * tan(θ), sinh(θ) / R]) #an arbitrary function
  function foo(AVec, BVec, original, novel)
    vofr = CovariantVector(f, original, true) # v of r
    zofu = CovariantVector(U->f(c⁻¹(h(U))), novel, true) # z of u
    @test vofr(r, true) ≈ zofu(u, true) # the same thing
    vofu = convert(vofr, novel) # v of u
    @test vofr(r, true) ≈ vofu(u, true)
  end

  for (V1, V2, c1, c2) ∈ Iterators.product((Cov, Con),
                                           (Cov, Con),
                                           (c, c⁻¹),
                                           (h, h⁻¹))
    foo(V1, V2, c1, c2)
  end
end

