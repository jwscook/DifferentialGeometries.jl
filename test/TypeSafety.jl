using Test, LinearAlgebra

@testset "TypeSafety tests" begin
  numberofiterations = 10

  cart = CoordinateTransform([cart->cart[1]])

  @testset "Con to Cov to Con" begin
    cov0 = Covariant(cart, [x->x])
    @test typeof(cov0) == Covariant
    con1 = Contravariant(cov0)
    @test typeof(con1) == Contravariant
    cov2 = Covariant(con1)
    @test typeof(cov2) == Covariant
  end

  @testset "Cov to Con to Cov" begin
    con0 = Contravariant(cart, [x->x])
    @test typeof(con0) == Contravariant
    cov1 = Covariant(con0)
    @test typeof(cov1) == Covariant
    con2 = Contravariant(cov1)
    @test typeof(con2) == Contravariant
  end

end
