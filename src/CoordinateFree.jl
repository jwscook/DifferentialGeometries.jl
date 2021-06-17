struct ∇{T}
  operand::T
  string::String
end

struct ∇o{T}
  operand::T
  string::String
end

struct ∇x{T}
  operand::T
  string::String
end

String(x::∇o) = "∇o" * x.string
String(x::∇x) = "∇×" * x.string

(g::∇)(x) = CovariantVector(x->grad(g.operand, x), true)
(c::∇x)(x) = ContravariantVector(x->curl(c.operand, x), true)
(d::∇o)(v::VectorField) = Scalar(dot(d.operand, v))
(d::∇o)(c::∇x) = 0
(c::∇x)(g::∇) = zeros(3)

Base.show(io::IO, x::∇o) = print(io, String(x))
Base.show(io::IO, x::∇x) = print(io, String(x))

