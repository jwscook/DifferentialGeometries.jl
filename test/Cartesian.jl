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
