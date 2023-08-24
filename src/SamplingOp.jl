export vectorizePattern

function LinearOperatorCollection.SamplingOp(::Type{T};
  pattern::P, shape::Tuple=()) where {P} where T <: Number
  if length(shape) == 0
    return SamplingOpImpl(pattern)
  else
    return SamplingOpImpl(pattern, shape, T)
  end
end


"""
 idx contains sampling index (for the first dimension) of a multidimensional Array
 of size "shape". Transform this into idx into the corresponding vector index
"""
function vectorizePattern(idx::T, shape::Tuple) where T<:AbstractArray{Int}
  return [ floor(Int,(i-1)/size(idx,1))*shape[1]+idx[i] for i = 1:length(idx) ]
end

"""
  SamplingOp(pattern::Array{Int}, shape::Tuple)

builds a `LinearOperator` which only returns the vector elements at positions
indicated by pattern.

# Arguents
* `pattern::Array{Int}` - indices to sample
* `shape::Tuple`        - size of the array to sample
"""
function SamplingOpImpl(pattern::T, shape::Tuple, type::Type=ComplexF64) where T<:AbstractArray{Int}
  ndims(pattern)>1 ?  idx = vectorizePattern(pattern, shape) : idx = pattern
  return opEye(type,length(idx))*opRestriction(idx, prod(shape))
end

function SamplingOpImpl(pattern::T) where T<:AbstractArray{Bool}

  function prod!(res::Vector{U}, x::Vector{V}) where {U,V}
    res .= pattern.*x
  end

  return LinearOperator(T, length(pattern), length(pattern), true, false, prod!)
end