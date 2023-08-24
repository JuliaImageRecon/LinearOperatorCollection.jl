function LinearOperatorCollection.constructLinearOperator(::Type{Op};
  weights, rep::Int=1) where Op <: WeightingOp{T} where T <: Number
  return WeightingOpImpl(weights, rep)
end

"""
  WeightingOp(weights::Vector{T}, rep::Int=1) where T

generates a `LinearOperator` which multiplies an input vector index-wise with `weights`

# Arguments
* `weights::Vector{T}` - weights vector
* `rep::Int=1`         - number of sub-arrays that need to be multiplied with `weights`
"""
function WeightingOpImpl(weights::T, rep::Int=1) where T<:AbstractVector
  weights_cat = repeat(weights,rep)
  return opDiagonal(weights_cat)
end
