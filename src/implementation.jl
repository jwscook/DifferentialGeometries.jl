using ForwardDiff, LinearAlgebra, NLsolve
using ForwardDiff: Dual, Tag, tagtype, value, partials


# η = ηₐ ∇a + ηᵦ ∇β + ηᵧ∇γ
# we want η to be a vector variable
# here we want ηi ui to be the ith component of η
# we want ηi to be a scalar variable
# ui needs to be a Co- or Contra-variant vector

# Covariant components Aᵢ 
# Contravariant components Aⁱ

abstract type Variance end
struct Covariant <: Variance end
struct Contravariant <: Variance end

abstract type AbstractCoordinateTransform{V<:Variance, D} end
const Act = AbstractCoordinateTransform # short hand

struct CoordinateTransform{V<:Variance, D, F} <: AbstractCoordinateTransform{V, D}
  f::F
end
(ct::CoordinateTransform)(u) = ct.f(u)
Base.:\(c::Act, x) = invert(c, x)
Base.length(c::CoordinateTransform{<:Variance, D}) where {D} = D

const ToCartesian = CoordinateTransform{Covariant}
const FromCartesian = CoordinateTransform{Contravariant}
ToCartesian(f::F, D::Integer) where {F} = CoordinateTransform{Covariant, D, F}(f)
FromCartesian(f::F, D::Integer) where {F} = CoordinateTransform{Contravariant, D, F}(f)

abstract type AbstractScalar{V<:Variance, D} <: Function end
struct Scalar{V<:Variance, D, F} <: AbstractScalar{V, D}
  f::F
  c::CoordinateTransform{V, D}
end
(s::Scalar)(x) = s.f(x)

J⃗(c::Act{Covariant}, r, ::Type{Covariant}) = ForwardDiff.jacobian(c, r)
J⃗(c::Act{Covariant}, r, ::Type{Contravariant}) = inv(J⃗(c, r, Covariant))
J⃗(c::Act{Contravariant}, r, ::Type{Covariant}) = ForwardDiff.jacobian(c.f, c \ r)
J⃗(c::Act{Contravariant}, r, ::Type{Contravariant}) = inv(J⃗(c, r, Covariant))
J⃗(c::Act{V}, x) where {V<:Variance} = J⃗(c, x, V)

covariantbasis(c::Act, x) = J⃗(c, x, Covariant)
contravariantbasis(c::Act, x) = J⃗(c, x, Contravariant)

gᵢⱼ(c::Act, x) = (j = J⃗(c, x); return j' * j)
gⁱʲ(c::Act, x) = inv(gᵢⱼ(c, x))

J(f, x) = sqrt(det(gᵢⱼ(f, x)))
jacobi(f, x) = J⃗(f, x)
jacobian(f, x) = J(f, x)

valueise(x) = x
valueise(x::T) where {T<:Dual} = valueise(value(x))
changevalue(x, y, s) = x == s ? y : x
function changevalue(x::AbstractVector, y, s)
  for i in eachindex(x)
    x[i] = changevalue(x[i], y[i], s[i])
  end
  return x
end
function changevalue(x::AbstractVector{<:Dual}, y, s)
  for i in eachindex(x)
    x[i] = changevalue(x[i], y[i], s[i])
  end
  return x
end
function changevalue(x::Dual{<:Tag}, y, s) where {T<:AbstractFloat, D}
  return Dual{tagtype(x)}(changevalue(x.value, y, s), partials(x))
end
function changevalue(x::Dual{<:Tag{<:Function,T}, T, D}, y::T, s::T
                     ) where {T<:AbstractFloat, D}
  return x.value == s ? Dual{tagtype(x)}(y, partials(x)) : x
end
changevalue(x, y) = changevalue(deepcopy(x), y, valueise.(x))
"""
Given coordinate transform f(y) -> x, find y for given x.
"""
function invert(f::F, x::AbstractVector{T}, initial_y=ones(size(x))
    ) where {F, T}
  f!(m, y) = (m .= f(y) .- valueise.(x); return nothing)
  y = NLsolve.nlsolve(f!, initial_y, autodiff=:forward, ftol=1e-12).zero
  # What is the proper way? This only works for a single AD pass.
  return changevalue(x, y)
end


invert(f::F) where {F} = x -> invert(f, x)
function invert(f::Act{V,D}) where {V<:Variance, D}
  f⁻¹(x) = invert(f, x)
  return CoordinateTransform{convert(V),D,typeof(f⁻¹)}(f⁻¹)
end

struct VectorField{V<:Variance, D, W<:Variance, F,
                   C<:AbstractCoordinateTransform{<:Variance, D}}
  f::F
  c::C
end

const ContravariantVector = VectorField{Contravariant}
const CovariantVector = VectorField{Covariant}

J⃗(v::VectorField, x, ::Type{V}) where {V<:Variance} = J⃗(v.c, x, V)
dl(c::AbstractCoordinateTransform, x, ::Type{Covariant}) = sqrt.(diag(gⁱʲ(c, x)))
dl(c::AbstractCoordinateTransform, x, ::Type{Contravariant}) = sqrt.(diag(gᵢⱼ(c, x)))
dl(v::VectorField{V}, x) where {V<:Variance} = dl(v.c, x, V)

function VectorField{V}(f::F, c::C, construct_from_unit_basis=false) where {
    V<:Variance, D, W<:Variance, F, C<:AbstractCoordinateTransform{W, D}}
  g(x) = construct_from_unit_basis ? f(x) ./ dl(c, x, V) : f(x)
  return VectorField{V, D, W, typeof(g), C}(g, c)
end
(v::VectorField)(x) = v.f(x)

(v::VectorField)(x, normalise::Bool) = normalise ? v(x) .* dl(v, x) : v(x)

function ContravariantVector(Aⱼ::CovariantVector)
  Aⁱ(x) = gⁱʲ(Aⱼ, x) * Aⱼ(x)
  return ContravariantVector(Aⁱ, Aⱼ.c)
end
function CovariantVector(Aʲ::ContravariantVector)
  Aᵢ(x) = gᵢⱼ(Aʲ, x) * Aʲ(x)
  return CovariantVector(Aᵢ, Aʲ.c)
end
CovariantVector(v::CovariantVector) = v
ContravariantVector(v::ContravariantVector) = v

import Base: convert
convert(::Type{Contravariant}) = Covariant
convert(::Type{Covariant}) = Contravariant
convert(Aⁱ::ContravariantVector) = CovariantVector(Aⁱ)
convert(Aᵢ::CovariantVector) = ContravariantVector(Aᵢ)
convert(v::VectorField{V,D,W,F,C}, c::C) where {V,D,W,F,C} = v

const Cov = Covariant
const Con = Contravariant
changecoords(a::Act{Cov,D}, b::Act{Cov,D}) where D = r -> a \ b(r)
changecoords(a::Act{Con,D}, b::Act{Cov,D}) where D = r -> a(b(r))
changecoords(a::Act{Cov,D}, b::Act{Con,D}) where D = r -> a \ (b \ r)
changecoords(a::Act{Con,D}, b::Act{Con,D}) where D = r -> a(b \ r)
function convert(v::VectorField{V,D,<:Variance}, c::Act{<:Variance,D}
    ) where {V<:Variance, D}
  return VectorField{V}(r -> v(changecoords(v.c, c)(r), true), c, true)
end

length(v::VectorField{<:Variance, D}) where {D} = D

dV(a) = J(a)
dV(a, b) = J(a, b)

function dS(c::Act{<:Variance, 3})
  function f(x)
    g_ij = gᵢⱼ(c, x)
    g(n, m) = sqrt(g_ij[n, n] * g_ij[m, m] - g[n, m]^2)
    return [g(filter(i -> i != k, 1:3)...) for k ∈ 1:3]
  end
  return CovariantVector(f, c, false)
end

for op ∈ (:J, :gⁱʲ, :gᵢⱼ)
  @eval $op(v::VectorField, x) = $op(v.c, x)
end

function ∇(f::T, c::Act, construct_from_unit_basis=false) where {T}
  return ContravariantVector(x->∇(f, x), c, construct_from_unit_basis)
end
∇(f::T, x) where {T} = ForwardDiff.jacobian(f, x)
∂(f::T, x) where {T} = inv(∇(f, x))

∇ᵢⱼ(v::VectorField, x) = ForwardDiff.jacobian(f, x)
∇ⁱʲ(v::VectorField, x) = inv(ForwardDiff.jacobian(f, x))

for op in (:+, :-)
  Vs = (ContravariantVector, CovariantVector)
  for (V1, V2) ∈ (Vs, reverse(Vs))
    @eval function Base.$(op)(a::$V1, b::$V2)
      @assert a.c == b.c
      return $V1(x->$op(a(x), $V1(b(x))), a.c)
    end
  end
end
import LinearAlgebra: dot, cross
dot(Aⁱ::ContravariantVector, Bᵢ::CovariantVector) = x->dot(Aⁱ(x), Bᵢ(x))
dot(Aᵢ::CovariantVector, Bⁱ::ContravariantVector) = dot(Bⁱ, Aᵢ)
dot(A::T, B::T) where {V, T<:VectorField{V}} = dot(A, convert(B))
function cross(Aⁱ::ContravariantVector, Bⁱ::ContravariantVector)
  @assert Aⁱ.c == Bⁱ.c
  return CovariantVector(x -> cross(Aⁱ(x), Bⁱ(x)) ./ J(Aⁱ, x), Aⁱ.c)
end
function cross(Aᵢ::CovariantVector, Bᵢ::CovariantVector)
  @assert Aᵢ.c == Bᵢ.c
  return ContravariantVector(x -> cross(Aᵢ(x), Bᵢ(x)) ./ J(Aᵢ, x), Aᵢ.c)
end
cross(A::T, B::T) where {V, T<:VectorField{V}} = cross(A, convert(B))

"""
grad(ϕ) = uⁱ (∂ᵢ ϕ)
"""
function grad(ϕ::F, c::Act{Covariant}) where {F}
  return CovariantVector(x->ForwardDiff.gradient(ϕ, x), c, false)
end
function grad(ϕ::F, c::Act{Contravariant}) where {F}
  return CovariantVector(x->ForwardDiff.gradient(ϕ, c \ x), c, false)
end
grad(s::Scalar) = grad(s.f, s.c)

import Base.div
"""
div(A⃗) = ∂ᵢ(J * Aⁱ) / J
"""
div(v::ContravariantVector) = Scalar(x->div(v, x), v.c)
function div(v::ContravariantVector, r)
  # v.c is CoordinateTransform{Covariant, D}
  return sum(diag(ForwardDiff.jacobian(y->J(v, y) * v(y), r))) / J(v, r)
end
div(v::CovariantVector) = div(ContravariantVector(v))
div(v::CovariantVector, x) = div(ContravariantVector(v), x)

∇o(a) = div(a)
∇o(a, b) = div(a, b)

#"""
#curl(A⃗) = ∑ₖ e⃗ₖ (∂ᵢAⱼ - ∂ⱼAᵢ) / J
#"""
#function curl(v::CovariantVector{3}, x::T) where {T}
#  #∂ⱼAᵢ = ∇(v, x, Covariant)
#  ∂ⱼAᵢ = ForwardDiff.jacobian(v, x)
#  ϵijk∂ⱼAᵢ = [∂ⱼAᵢ[3,2] - ∂ⱼAᵢ[2,3], ∂ⱼAᵢ[1,3] - ∂ⱼAᵢ[3,1], ∂ⱼAᵢ[2,1] - ∂ⱼAᵢ[1,2]]
#  return ϵijk∂ⱼAᵢ ./ J(v, x)
#end
#
#curl(v::CovariantVector) = ContravariantVector(x->curl(v, x), v.c)
#curl(f::F, c::Act) where {F} = curl(CovariantVector(f, c, true))
#curl(v::ContravariantVector) = curl(CovariantVector(v))
#curl(v::ContravariantVector, x) = curl(CovariantVector(v), x)

"""
curl(A⃗) = ∑ₖ e⃗ₖ (∂ᵢAⱼ - ∂ⱼAᵢ) / J
"""
function curl(v::CovariantVector{3}) where {T}
  function inner(x)
    ∂ⱼAᵢ = ForwardDiff.jacobian(v, x)
    ϵijk∂ⱼAᵢ = [∂ⱼAᵢ[3,2] - ∂ⱼAᵢ[2,3], ∂ⱼAᵢ[1,3] - ∂ⱼAᵢ[3,1], ∂ⱼAᵢ[2,1] - ∂ⱼAᵢ[1,2]]
    return ϵijk∂ⱼAᵢ ./ J(v, x)
  end
  return ContravariantVector(inner, v.c)
end
curl(v::ContravariantVector) = curl(CovariantVector(v))
curl(v::ContravariantVector, x) = curl(CovariantVector(v))(x)

#curl(v::CovariantVector) = ContravariantVector(x->curl(v, x), v.c)
#curl(f::F, c::Act) where {F} = curl(CovariantVector(f, c, true))


∇x(a) = curl(a)
∇x(a, b) = curl(a, b)

vectorlaplacian(v::CovariantVector) = grad(div(v)) - curl(curl(v))
vectorlaplacian(v::CovariantVector, x) = grad(div(v))(x) - curl(curl(v))(x)

