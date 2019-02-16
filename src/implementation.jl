using ForwardDiff, LinearAlgebra, Combinatorics, NLsolve, Optim

import Base.length
import LinearAlgebra.dot
import LinearAlgebra.cross

abstract type AbstractTransform end

struct CoordinateTransform{T<:Function} <: AbstractTransform
  f::T
  dims::Int
  function CoordinateTransform(fx::Vector{T}) where {T}
    return new{T}(x -> [f(x) for f ∈ fx], length(fx))
  end
end
const CT = CoordinateTransform
length(c::CT) = c.dims
(c::CT)(x, i::Integer) = c.f(x)[i]
(c::CT)(x) = c.f(x)
Base.:\(c::CT, x) = inverse(c, x)


#function solve(f::Function, dims, ic=rand(dims);
#     rtol=2*eps(), atol=eps())
#  options = Optim.Options(x_tol=rtol, f_tol=rtol, g_tol=rtol,
#    allow_f_increases=true)
#  result = Optim.optimize(f, ic, Newton(), options)
#  return Optim.minimizer(result)
#end
#function inverse(fs::Vector{T}, coords::AbstractVector) where {T<:Function}
#    g(x) = [f(x) for f ∈ fs]
#  return solve(x -> sum((coords .- g(x)).^2), length(fs))
#end
#function inverse(c::CT, coords::AbstractVector{T}) where {T}
#  return solve(x -> sum((coords .- c(x)).^2), c.dims)
#end

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
"""An Co/Contravariant basis vector"""
struct BasisVector{T<:Component}
  fs::Vector{T}
end
const BV = BasisVector
Base.iterate(it::BasisVector, x) = iterate(it.fs, x)
(A::BasisVector)(x) = hcat((f(x) for f ∈ A.fs)...)

"""Contravariant component Aⁱ"""
struct Contravariant{T<:Function} <: Component
  f::T
end
const Con = Contravariant
(Aⁱ::Contravariant)(x) = Aⁱ.f(x)

"""Covariant component, Aᵢ """
struct Covariant{T<:Function} <: Component
  f::T
end
const Cov = Covariant
(Aᵢ::Covariant)(x) = Aᵢ.f(x)

∇(f::T) where {T<:Function} = x -> ForwardDiff.gradient(f, x)
∇(c::CT, i::Integer) = Cov(x -> ForwardDiff.gradient(y -> c(y, i), x))
∇(c::CT) = BV([∇(c, i) for i ∈ 1:c.dims])
∂(c::CT, f::Function) = x -> inv(∇(c)(x)) * ∇(f)(x)
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

function div(c::CT, Aⁱ::Contravariant)
  ∂iJAⁱ(x, i) = ForwardDiff.gradient(y -> Aⁱ(y) * J(c)(y), x)
  ∂JAⁱ(x) = sum(∂iJAi(x, i) for i in 1:c.dims)
  return x -> 1/J(a)(x) * ∂JA(x)
end
div(c::CT, Aᵢ::Cov) = div(c, Con(c, Aᵢ))
div(c::CT, A::BV) = x -> mapreduce(Ai -> div(c, Ai)(x), +, A)

function curl(c::CT, Aᵢ::Cov)
  itr = Combinatorics.permutations(1:c.dims, 3)
  return x -> mapreduce(i->levicevita(i)*∂uⁱ(c, Aᵢ.f)(x)[i[3]], +, itr) / J(a)(x)
end
curl(c::CT, Aⁱ::Con) = curl(c, Cov(c, Aⁱ))
curl(c::CT, A::BV) = x -> mapreduce(Ai -> curl(c, Ai)(x), +, A)



