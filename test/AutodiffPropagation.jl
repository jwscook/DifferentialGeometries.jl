@testset "Autodiff propagation" begin
  f = ToCartesian(r->[r[1] * cos(r[2]), r[1]*sin(r[2])], 2)
  f⁻¹ = FromCartesian(x->[sqrt(x[1]^2 + x[2]^2), atan(x[2], x[1])], 2)

  X = rand(2) .- 0.5
  R = f⁻¹(X)

  @test ForwardDiff.jacobian(f, R) ≈ inv(ForwardDiff.jacobian(f⁻¹, X))
  @test inv(ForwardDiff.jacobian(f, R)) ≈ ForwardDiff.jacobian(f⁻¹, X)
  @test X ≈ f⁻¹ \ R
  @test inv(ForwardDiff.jacobian(f, R)) ≈ ForwardDiff.jacobian(r-> f⁻¹(f⁻¹ \ r), R)
end


