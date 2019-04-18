module DifferentialGeometries

include("implementation.jl")

export AbstractCoordinateSystem, CoordinateSystem
export AbstractScalar, Scalar
export Covariant, Contravariant
export ToCartesian, FromCartesian
export CovariantVector, ContravariantVector
export ∇, ∂, grad, div, ∇o, curl, ∇x, J, dV, dl, gᵢⱼ, gⁱʲ, J⃗
export invert

end
