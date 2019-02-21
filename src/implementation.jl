using ForwardDiff, LinearAlgebra, Combinatorics, NLsolve, Optim

import Base.length
import Base.iterate
import Base.eachindex
import Base.enumerate
import Base.getindex
import LinearAlgebra.dot
import LinearAlgebra.cross

abstract type AbstractTransform end

struct CoordinateTransform{T<:Function} <: AbstractTransform
  f::T
  dims::Int
end
function CoordinateTransform(fx::Vector{Function}) where {T}
  return CoordinateTransform(x -> [f(x) for f ∈ fx], length(fx))
end
const CT = CoordinateTransform
length(c::CT) = c.dims
(c::CT)(x, i::Integer) = c.f(x)[i]
(c::CT)(x) = c.f(x)
Base.:\(c::CT, x) = inverse(c, x)

function solve(f::Function, dims::Integer; ic::AbstractVector=ones(dims),
    rtol=2*eps(), atol=eps())
  result = NLsolve.nlsolve(f, ic, autodiff=:forward)
  return result.zero
end
function inverse(fs::Vector{T}, coords::AbstractVector) where {T<:Function}
  g(x) = [f(x) for f ∈ fs]
  return solve(x -> g(x) .- coords, length(fs))
end
function inverse(c::CT, coords::AbstractVector{T}) where {T}
  return solve(x -> c(x) .- coords, c.dims)
end

abstract type Component end

"""Contravariant component Aⁱ"""
struct Contravariant <: Component
  f::Function
end
const Con = Contravariant
(Aⁱ::Contravariant)(x) = Aⁱ.f(x)

"""Covariant component, Aᵢ """
struct Covariant <: Component
  f::Function
end
const Cov = Covariant
(Aᵢ::Covariant)(x) = Aᵢ.f(x)

"""A Co/Contravariant basis vector"""
struct BasisVector{T<:Component}
  Ai::Vector{T}
end
const BV = BasisVector
Base.iterate(b::BV) = iterate(b.Ai)
Base.iterate(b::BV, x) = iterate(b.Ai, x)
Base.length(b::BV) = length(b.Ai)
Base.eachindex(b::BV) = eachindex(b.Ai)
Base.enumerate(b::BV) = enumerate(b.Ai)
Base.getindex(b::BV, i::Integer) = b.Ai[i]

(A::BasisVector)(x) = return hcat((f(x) for f ∈ A.Ai)...)
#(A::BV{Con})(x) = vcat((f(x)' for f ∈ A.Ai)...)
#(A::BasisVector{Cov{T}})(x) where {T} = hcat((f(x) for f ∈ A.Ai)...)
#(A::BasisVector{Con{T}})(x) where {T} = vcat((f(x)' for f ∈ A.Ai)...)

∇(f::T) where {T<:Function} = x -> ForwardDiff.gradient(f, x)
∇(c::CT, i::Integer) = Cov(x -> ForwardDiff.gradient(y -> c(y, i), x))
∇(c::CT) = BV([∇(c, i) for i ∈ 1:c.dims])
∂(c::CT, f::Function) = x -> inv(∇(c)(x)) * ∇(f)(x)
∂(c::CT, Aᵢ::Cov) = x -> inv(∇(c)(x)) * ∇(y->Aᵢ(y))(x)
#eⁱ = ∇
#eᵢ = ∂

gⁱʲ(a::CT, b::CT, i::Integer, j::Integer) = x -> dot(∇(a, i)(x), ∇(b, j)(x))
gⁱʲ(a::CT, b::CT=a) = x -> ∇(a)(x)' * ∇(b)(x)

gᵢⱼ(a::CT, b::CT=a) = x -> inv(gⁱʲ(a, b)(x))

J(a::CT, b::CT=a) = x -> sqrt(det(gᵢⱼ(a, b)(x)))

Con(c::CT, Aᵢ::Cov) = x -> gⁱʲ(c)(x) * Aᵢ(x)
Cov(c::CT, Aⁱ::Con) = x -> gᵢⱼ(c)(x) * Aⁱ(x)

BV(c::CT, Aᵢs::BV{Cov}) = BV([Con(c, Aᵢ) for Aᵢ ∈ Aᵢs])
BV(c::CT, Aⁱs::BV{Con}) = BV([Cov(c, Aⁱ) for Aⁱ ∈ Aⁱs])

function div(c::CT, Aⁱ::Contravariant, i::Integer)
  return x -> ∂(c, y -> Aⁱ(y) * J(c)(y))(x)[i] / J(c)(x)
end
div(c::CT, Aᵢ::Cov) = div(c, Con(c, Aᵢ))
div(c::CT, b::BV) = x -> mapreduce(i -> div(c, b.Ai[i], i)(x), +, eachindex(b))
div(c::CT, fs::Vector{T}) where {T<:Function} = div(c, BV(Con.(fs)))

function curl(c::CT, b::BV{Cov})
  @assert c.dims == 3
  f1(x) = (∂(c, b[3])(x)[2] - ∂(c, b[2])(x)[3]) / J(c)(x)
  f2(x) = (∂(c, b[1])(x)[3] - ∂(c, b[3])(x)[1]) / J(c)(x)
  f3(x) = (∂(c, b[2])(x)[1] - ∂(c, b[1])(x)[2]) / J(c)(x)
  return BV(Con.([f1, f2, f3]))
end
curl(c::CT, Aⁱ::BV{Con}) = curl(c, BV(c, Aⁱ))
curl(c::CT, fs::Vector{T}) where {T<:Function} = curl(c, BV(Cov.(fs)))

function dot(c::CT, a::BV{Con}, b::BV{Cov})
  error("Untested")
  return x -> mapreduce(a.Ai(x) * b.Ai(), +, 1:c.dims)
end
dot(c::CT, a::BV{Cov}, b::BV{Con}) = dot(c, b, a)
dot(c::CT, a::BV{Con}, b::BV{Con}) = dot(c, a, BV(c, b))
dot(c::CT, a::BV{Cov}, b::BV{Cov}) = dot(c, a, BV(c, b))

