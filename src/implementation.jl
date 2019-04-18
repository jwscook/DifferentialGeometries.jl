using ForwardDiff, LinearAlgebra, NLsolve


# η = η⊥ ∇Ψ + η∧ ∇χ + ηb ∇ξ
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

J⃗(c::Act, x, ::Type{Covariant}) = ForwardDiff.jacobian(c, x)
J⃗(c::Act, x, ::Type{Contravariant}) = inv(ForwardDiff.jacobian(c, x))
J⃗(c::Act{V}, x) where {V<:Variance} = J⃗(c, x, V)

covariantbasis(c::Act, x) = J⃗(c, x, Covariant)
contravariantbasis(c::Act, x) = J⃗(c, x, Contravariant)

∇(f, x, ::Type{Covariant}) = ForwardDiff.jacobian(f, x)
∇(f, x, ::Type{Contravariant}) = inv(ForwardDiff.jacobian(f, x))

gᵢⱼ(c::AbstractCoordinateTransform, x) = (j = J⃗(c, x); return j' * j)
gⁱʲ(c::AbstractCoordinateTransform, x) = inv(gᵢⱼ(c, x))

J(f, x) = sqrt(det(gᵢⱼ(f, x)))
jacobi(f, x) = J⃗(f, x)
jacobian(f, x) = J(f, x)

function invert(f::F, x, initial_x=ones(size(x))) where {F}
  f!(m, y) = (m .= f(y) .- x; return nothing)
  y = NLsolve.nlsolve(f!, initial_x, autodiff=:forward)
  return y.zero
end

invert(f::F) where {F} = x -> invert(f, x)

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
Base.convert(Aⁱ::ContravariantVector) = CovariantVector(Aⁱ)
Base.convert(Aᵢ::CovariantVector) = ContravariantVector(Aᵢ)
Base.length(v::VectorField{<:Variance, D}) where {D} = D

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
  return CovariantVector(x->J⃗(c, x)' * ForwardDiff.gradient(ϕ, x), c, false)
end
grad(s::Scalar) = grad(s.f, s.c)

import Base.div
"""
div(A⃗) = ∂ᵢ(J * Aⁱ) / J
"""
div(v::ContravariantVector) = Scalar(x->div(v, x), v.c)
function div(v::ContravariantVector{D, <:Covariant}, x) where {D}
  # v.c is CoordinateTransform{Covariant, D}
  return sum(diag(∇(y->J(v, y) * v(y), x, Covariant))) / J(v, x)
end
function div(v::ContravariantVector{D, <:Contravariant}, x) where {D}
  # v.c is CoordinateTransform{Contravariant, D}
  return sum(diag(∇(x->v(x) * J(v.c, x), x, Covariant) * J⃗(v, x, Contravariant))) / J(v, x)
end

div(v::CovariantVector) = div(ContravariantVector(v))
div(v::CovariantVector, x) = div(ContravariantVector(v), x)

∇o(a) = div(a)
∇o(a, b) = div(a, b)

"""
curl(A⃗) = ∑ₖ e⃗ₖ (∂ᵢAⱼ - ∂ⱼAᵢ) / J
"""
function curl(v::CovariantVector{3, <:Covariant}, x::T) where {T}
  ∂ⱼAᵢ = ∇(v, x, Covariant)
  ϵijk∂ⱼAᵢ = [∂ⱼAᵢ[3,2] - ∂ⱼAᵢ[2,3], ∂ⱼAᵢ[1,3] - ∂ⱼAᵢ[3,1], ∂ⱼAᵢ[2,1] - ∂ⱼAᵢ[1,2]]
  return ϵijk∂ⱼAᵢ ./ J(v, x)
end
function curl(v::CovariantVector{3, <:Contravariant}, x::T) where {T}
  ∂ⱼAᵢ = ∇(v, x, Covariant) * J⃗(v, x, Contravariant)
  ϵijk∂ⱼAᵢ = [∂ⱼAᵢ[3,2] - ∂ⱼAᵢ[2,3], ∂ⱼAᵢ[1,3] - ∂ⱼAᵢ[3,1], ∂ⱼAᵢ[2,1] - ∂ⱼAᵢ[1,2]]
  return ϵijk∂ⱼAᵢ ./ J(v, x)
end

curl(v::CovariantVector) = ContravariantVector(x->curl(v, x), v.c)
curl(f::F, c::Act) where {F} = curl(CovariantVector(f, c, true))
curl(v::ContravariantVector) = curl(CovariantVector(v))
curl(v::ContravariantVector, x) = curl(CovariantVector(v), x)

∇x(a) = curl(a)
∇x(a, b) = curl(a, b)

vectorlaplacian(v::CovariantVector) = grad(div(v)) - curl(curl(v))
vectorlaplacian(v::CovariantVector, x) = grad(div(v))(x) - curl(curl(v))(x)

