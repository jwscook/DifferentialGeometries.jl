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

abstract type Component{T} end

"""Contravariant component Aⁱ"""
struct Contravariant{T<:Function} <: Component{T}
  f::T
end
const Con = Contravariant

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
#∇(c::CT) = Cov(c, [∇(y -> c(y, i)) for i ∈ 1:c.dims])
∇(c::CT) = Cov(c, [x -> 1 for i ∈ 1:c.dims])
=======
"""Covariant component, Aᵢ"""
struct Covariant{T<:Function} <: Component{T}
  f::T
end
const Cov = Covariant
(Aᵢ::Covariant)(x) = Aᵢ.f(x)

"""
A Co/Contravariant basis vector
  A = Σᵢ Aⁱeᵢ =  Σᵢ Aᵢeⁱ
"""
struct VectorSpace{T<:Component}
  Ai::Vector{T}
end
#VS(c::CT, fs::Vector{Function}) = VS([Cov(c, f) for f ∈ fs])
const VS = VectorSpace
Base.iterate(b::VS) = iterate(b.Ai)
Base.iterate(b::VS, x) = iterate(b.Ai, x)
Base.length(b::VS) = length(b.Ai)
Base.eachindex(b::VS) = eachindex(b.Ai)
Base.enumerate(b::VS) = enumerate(b.Ai)
Base.getindex(b::VS, i::Integer) = b.Ai[i]

#(A::VectorSpace)(x) = hcat((f(x) for f ∈ A.Ai)...)
function (A::VectorSpace)(x)
  return hcat((f(x) for f ∈ A.Ai)...)
end
#(A::VS{Con})(x) = vcat((f(x)' for f ∈ A.Ai)...)
#(A::VectorSpace{Cov{T}})(x) where {T} = hcat((f(x) for f ∈ A.Ai)...)
#(A::VectorSpace{Con{T}})(x) where {T} = vcat((f(x)' for f ∈ A.Ai)...)

∇(f::T) where {T<:Function} = x -> ForwardDiff.gradient(f, x)
∇(c::CT, i::Integer) = Cov(x -> ForwardDiff.gradient(y -> c(y, i), x))
∇(c::CT) = VS([∇(c, i) for i ∈ 1:c.dims])
>>>>>>> master
∂(c::CT, f::Function) = x -> inv(∇(c)(x)) * ∇(f)(x)
#∂(Aᵢ::Cov) = x -> inv(∇(Aᵢ.ct)(x)) * ∇(y->Aᵢ(y))(x)
∂(Aᵢ::Cov, i::Integer) = x -> inv(∇(Aᵢ.ct)(x)) * ∇(y->Aᵢ[i](y))(x)
#eⁱ = ∇
#eᵢ = ∂

gⁱʲ(a::CT, b::CT, i::Integer, j::Integer) = x -> dot(∇(a, i)(x), ∇(b, j)(x))
gⁱʲ(a::CT, b::CT=a) = x -> ∇(a)(x)' * ∇(b)(x)

gᵢⱼ(a::CT, b::CT=a) = x -> inv(gⁱʲ(a, b)(x))

J(a::CT, b::CT=a) = x -> sqrt(det(gᵢⱼ(a, b)(x)))

Con(c::CT, Aᵢ::Cov) = x -> gⁱʲ(c)(x) * Aᵢ(x)
Cov(c::CT, Aⁱ::Con) = x -> gᵢⱼ(c)(x) * Aⁱ(x)

VS(c::CT, Aᵢs::VS{Cov}) = VS([Con(c, Aᵢ) for Aᵢ ∈ Aᵢs])
VS(c::CT, Aⁱs::VS{Con}) = VS([Cov(c, Aⁱ) for Aⁱ ∈ Aⁱs])

function div(Aⁱ::Contravariant, i::Integer)
  return x -> (∂(Aⁱ.ct, y -> Aⁱ[i](y)[i] * J(Aⁱ.ct)(y))(x) / J(Aⁱ.ct)(x))[i]
end
div(c::CT, Aᵢ::Cov) = div(c, Con(c, Aᵢ))
div(c::CT, b::VS) = x -> mapreduce(i -> div(c, b.Ai[i], i)(x), +, eachindex(b))
div(c::CT, fs::Vector{T}) where {T<:Function} = div(c, VS(Con.(fs)))

function curl(c::CT, Aᵢ::VS{Cov})
  @assert c.dims == 3
  f1(x) = (∂(c, Aᵢ[3])(x)[2] - ∂(c, Aᵢ[2])(x)[3])
  f2(x) = (∂(c, Aᵢ[1])(x)[3] - ∂(c, Aᵢ[3])(x)[1])
  f3(x) = (∂(c, Aᵢ[2])(x)[1] - ∂(c, Aᵢ[1])(x)[2])
  return Con(x -> [f1(x), f2(x), f3(x)] ./ J(c)(x))
end
curl(c::CT, Aⁱ::VS{Con}) = curl(c, VS(c, Aⁱ))
curl(c::CT, fs::Vector{T}) where {T<:Function} = curl(c, VS(Cov.(fs)))

function dot(c::CT, a::VS{Con}, b::VS{Cov})
dot(a::Cov, b::Con) = dot(b, a)
dot(a::Con, b::Con) = dot(a, Cov(b))
dot(a::Cov, b::Cov) = dot(a, Con(b))
dot(c::CT, a::VS{Cov}, b::VS{Con}) = dot(c, b, a)
dot(c::CT, a::VS{Con}, b::VS{Con}) = dot(c, a, VS(c, b))
dot(c::CT, a::VS{Cov}, b::VS{Cov}) = dot(c, a, VS(c, b))

