"""
    WeightingOp(::Type{T}; weights::Vector{T}, rep::Int=1) where T

generates a `LinearOperator` which multiplies an input vector index-wise with `weights`

# Arguments
* `weights::Vector{T}` - weights vector
* `rep::Int=1`         - number of sub-arrays that need to be multiplied with `weights`
"""
mutable struct WeightingOp{T, vecT <: AbstractVector{T}} <: AbstractLinearOperatorFromCollection{T}
  op::LinearOperator{T}
  weights::vecT
  function WeightingOp(weights::vecT, rep::Int=1) where {T <: Number, vecT<:AbstractVector{T}}
    weights_cat = repeat(weights,rep)
    return new{T, vecT}(opDiagonal(weights_cat), weights_cat)
  end
end
WeightingOp(::Type{T}; weights::vecT, rep::Int=1) where {T <: Number, vecT<:AbstractVector{T}} = WeightingOp(weights, rep)

function Base.getproperty(wop::WeightingOp, field::Symbol)
  if in(field, (:op, :weights)) 
    return getfield(wop, field)
  else
    return getproperty(getfield(wop, :op), field)
  end
end
Base.setproperty!(wop::WeightingOp, field::Symbol, value) = setproperty!(wop.op, field, value)

storage_type(wop::WeightingOp) = storage_type(wop.op)