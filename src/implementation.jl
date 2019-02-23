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
function CoordinateTransform(fx::Vector{T}) where {T}
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

"""
Covariant component, Aᵢ
A = Aᵢ ∇uⁱ
"""
struct Covariant <: Component
  ct::CT
  i::Vector
  function Covariant(c::CT, fs::Vector{T}) where {T<:Function}
    return new(c, [x -> f(x) * ∇(c, k)(x) for (k, f) ∈ enumerate(fs)])
  end
end
const Cov = Covariant
#(Aᵢ::Cov)(x) = Aᵢ.i(x)

"""Contravariant component Aⁱ
A = Aⁱ ∂uᵢ
"""
struct Contravariant <: Component
  ct::CT
  i::Vector
  function Contravariant(c::CT, fs::Vector{T}) where {T<:Function}
    return new(c, raiseindex(Covariant(c, fs)))
  end
end
const Con = Contravariant
#(Aⁱ::Con)(x) = Aⁱ.i(x)

raiseindex(A::Covariant) = [x -> gⁱʲ(A.ct)(x) * Aᵢ(x) for Aᵢ ∈ A]
lowerindex(A::Contravariant) = [x -> gᵢⱼ(A.ct)(x) * Aⁱ(x) for Aⁱ ∈ A]

"""A Co/Contravariant basis vector"""
Base.iterate(a::Component) = iterate(a.i)
Base.iterate(a::Component, x) = iterate(a.i, x)
Base.length(a::Component) = length(a.i)
Base.eachindex(a::Component) = eachindex(a.i)
Base.enumerate(a::Component) = enumerate(a.i)
Base.getindex(a::Component, i::Integer) = a.i[i]

(Aⁱ::Con)(x) = hcat((f(x) for f ∈ Aⁱ.i)...)
(Aᵢ::Cov)(x) = vcat((f(x)' for f ∈ Aᵢ.i)...)
# [∂x r, ∂x θ;
#  ∂y r, ∂y θ]

∇(f::T) where {T<:Function} = x -> ForwardDiff.gradient(f, x)
∇(c::CT, i::Integer) = x -> ForwardDiff.gradient(y -> c(y, i), x)
#∇(c::CT, i::Integer) = Cov(c, ∇(y -> c(y, i)))
∇(c::CT) = Cov(c, [∇(y -> c(y, i)) for i ∈ 1:c.dims])
∂(c::CT, f::Function) = x -> inv(∇(c)(x)) * ∇(f)(x)
∂(Aᵢ::Cov) = x -> inv(∇(Aᵢ.ct)(x)) * ∇(y->Aᵢ(y))(x)
#eⁱ = ∇
#eᵢ = ∂

gⁱʲ(a::CT, b::CT, i::Integer, j::Integer) = x -> dot(∇(a, i)(x), ∇(b, j)(x))
gⁱʲ(a::CT, b::CT=a) = x -> ∇(a)(x)' * ∇(b)(x)

gᵢⱼ(a::CT, b::CT=a) = x -> inv(gⁱʲ(a, b)(x))

J(a::CT, b::CT=a) = x -> sqrt(det(gᵢⱼ(a, b)(x)))

Cov(Aⁱ::Con) = Cov(Aⁱ.ct, lowerindex(Aⁱ))
Con(Aᵢ::Cov) = Con(Aᵢ.ct, raiseindex(Aᵢ))
norm(A::Cov) = x -> [abs(gⁱʲ(A.ct)(x)[k, k]) * Aᵢ(x)[k] for (k, Aᵢ) ∈ enumerate(A)]
norm(A::Con) = x -> [abs(gᵢⱼ(A.ct)(x)[k, k]) * Aⁱ(x)[k] for (k, Aⁱ) ∈ enumerate(A)]

function div(Aⁱ::Contravariant, i::Integer)
  return x -> ∂(Aⁱ.ct, y -> Aⁱ[i](y) * J(Aⁱ.ct)(y))(x)[i] / J(Aⁱ.ct)(x)
end
div(Aᵢ::Cov) = div(Con(Aᵢ))
div(Aⁱ::Con) = x -> mapreduce(i -> div(Aⁱ, i)(x), +, eachindex(Aⁱ))
div(ct::CT, fs::Vector{T}) where {T<:Function} = div(Con(ct, fs))

# Bᵏ = ∇×(Aᵢeⁱ) = ϵijk ∂i Aⱼ e_k
function curl(Aᵢ::Cov)
  @assert Aᵢ.ct.dims == 3
  f1(x) = (∂(Aᵢ.ct, Aᵢ[3])(x)[2] - ∂(Aᵢ.ct, Aᵢ[2])(x)[3]) / J(Aᵢ.ct)(x)
  f2(x) = (∂(Aᵢ.ct, Aᵢ[1])(x)[3] - ∂(Aᵢ.ct, Aᵢ[3])(x)[1]) / J(Aᵢ.ct)(x)
  f3(x) = (∂(Aᵢ.ct, Aᵢ[2])(x)[1] - ∂(Aᵢ.ct, Aᵢ[1])(x)[2]) / J(Aᵢ.ct)(x)
  return Con(Aᵢ.ct, [f1, f2, f3])
end
curl(Aⁱ::Con) = curl(Cov(Aⁱ))
curl(c::CT, fs::Vector{T}) where {T<:Function} = curl(Cov(c, fs))

function dot(a::Con, b::Cov)
  error("Untested")
  return x -> mapreduce(a.i(x) * b.i(x), +, 1:a.ct.dims)
end
dot(a::Cov, b::Con) = dot(b, a)
dot(a::Con, b::Con) = dot(a, Cov(b))
dot(a::Cov, b::Cov) = dot(a, Con(b))

