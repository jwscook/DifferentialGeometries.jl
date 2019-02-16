using ForwardDiff, LinearAlgebra, Combinatorics, Optim

import Base.length
import LinearAlgebra.dot
import LinearAlgebra.cross

struct CoordinateTransform{T<:Function}
  f::T
  dims::Int
  function CoordinateTransform(fx::Vector{T}) where {T}
    output = new{T}(x -> [f(x) for f ∈ fx], length(fx))
    return output
  end
end
const CT = CoordinateTransform
length(c::CT) = c.dims
(c::CT)(x, i::Integer) = c.f(x)[i]
(c::CT)(x) = c.f(x)

function solve(c::CT, f::Function, ic=rand(length(c));
    rtol=2*eps(), atol=eps())
  options = Optim.Options(x_tol=rtol, f_tol=rtol, g_tol=rtol,
    allow_f_increases=true)
  result = Optim.optimize(f, ic, Newton(), options)
  return Optim.minimizer(result)
end

function inverse(c::CT, coordinates::Vector{T}) where {T}
  flocal(x) = mapreduce(i -> sum((coordinates .- c(x)).^2), +, 1:length(c))
  return solve(c, flocal)
end

abstract type Component end
"""An Co/Contravariant basis vector"""
struct BasisVector{T<:Component}
  fs::Vector{T}
end
Base.iterate(it::BasisVector, x) = iterate(it.fs, x)
(A::BasisVector)(x) = hcat((f(x) for f ∈ A.fs)...)

"""Contravariant components Aⁱ"""
struct Contravariant{T<:Function} <: Component
  f::T
end
(Aⁱ::Contravariant)(x) = Aⁱ.f(x)

"""Covariant components, Aᵢ """
struct Covariant{T<:Function} <: Component
  f::T
end
(Aᵢ::Covariant)(x) = Aᵢ.f(x)

∇(c::CT, i::Integer) = Covariant(x -> ForwardDiff.gradient(y -> c(y, i), x))
∇(c::CT) = BasisVector(
  [Covariant(x -> ForwardDiff.gradient(y -> c(y, i), x)) for i ∈ 1:c.dims])
gⁱʲ(a::CT, b::CT, i::Integer, j::Integer) = x -> dot(∇(a, i)(x), ∇(b, j)(x))
gⁱʲ(a::CT, b::CT=a) = x -> ∇(a)(x)' * ∇(b)(x)

gᵢⱼ(a::CT, b::CT=a) = x->inv(gⁱʲ(a, b)(x))
J(a::CT, b::CT=a) = x -> sqrt(det(gᵢⱼ(a, b)(x)))

Covariant(c::CT, Aⁱ::Contravariant, j::Integer) = x -> dot(gᵢⱼ(c, j)(x), Aⁱ(x))
function Covariant(c::CT, Aⁱ::Contravariant)
  f = x -> mapreduce(j -> Convariant(c, Aⁱ, j)(x), +, 1:length(c))
  return Covariant(f)
end

Contravariant(c::CT, Aᵢ::Covariant, j::Integer) = x -> dot(gⁱʲ(c, j)(x), Aᵢ(x))
function Contravariant(c::CT, Aᵢ::Covariant)
  f = x -> mapreduce(j -> Contravariant(c, Aᵢ, j)(x), +, 1:length(c))
  return Contravariant(f)
end

function ∂uⁱ(c::CT, f::Function)
  g = x -> ForwardDiff.gradient(f, x)
  return x -> mapreduce(dot(gᵢⱼ(c)(x)[:, i], g(x)), +, 1:length(c))
end

function div(c::CT, Aⁱ::Contravariant)
  ∂iJAⁱ(x, i) = ForwardDiff.gradient(y -> Aⁱ(y) * J(c)(y), x)
  ∂JAⁱ(x) = sum(∂iJAi(x, i) for i in 1:length(c))
  return x -> 1/J(a)(x) * ∂JA(x)
end
div(c::CT, Aᵢ::Covariant) = div(c, Contravariant(c, Aᵢ))
div(c::CT, A::BasisVector) = x -> mapreduce(Ai -> div(c, Ai)(x), +, A)

function curl(c::CT, Aᵢ::Covariant)
  itr = Combinatorics.permutations(1:length(c), 3)
  return x -> mapreduce(i->levicevita(i)*∂uⁱ(c, Aᵢ.f)(x)[i[3]], +, itr) / J(a)(x)
end
curl(c::CT, Aⁱ::Contravariant) = curl(c, Covariant(c, Aⁱ))
curl(c::CT, A::BasisVector) = x -> mapreduce(Ai -> curl(c, Ai)(x), +, A)

function cross(c::CT, Aⁱ::Contravariant, Bʲ::Contravariant)
  error("not implemented")
end
function cross(c::CT, Aⁱ::Contravariant, Bⱼ::Covariant)
  error("not implemented")
end
function cross(c::CT, Aᵢ::Covariant, Bⱼ::Covariant)
  error("not implemented")
end
function cross(c::CT, Aⁱ::Covariant, Bʲ::Contravariant)
  error("not implemented")
end

function dot(c::CT, Aⁱ::Contravariant, Bʲ::Contravariant)
  error("not implemented")
end
function dot(c::CT, Aᵢ::Covariant, Bⱼ::Covariant)
  error("not implemented")
end
function dot(c::CT, Aⁱ::Contravariant, Bⱼ::Covariant)
  error("not implemented")
end
dot(c::CT, Aⁱ::Covariant, Bʲ::Contravariant) = dot(Bʲ, Aᵢ)





