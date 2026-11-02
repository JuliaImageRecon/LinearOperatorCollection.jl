export vectorizePattern

"""
    SamplingOp(::Type{T}; pattern, shape::Tuple=(), S = Vector{T})

builds a linear operator that samples a vector at the positions given by `pattern`.

Two modes are supported:
* integer indices with `shape`: returns the entries of a vectorized array of size
  `shape` at the indices in `pattern`. A multidimensional `pattern` holds indices
  into the first dimension, one column per remaining index, and is converted with
  `vectorizePattern`. A one-dimensional `pattern` is used as is and not copied.
* Boolean mask without `shape`: multiplies the input elementwise with a copy of
  `pattern` stored on the backend of `S`.

`Adapt.adapt(to, op)` rebuilds the operator for the storage type `to` from its
effective pattern. Integer indices are copied and stay host arrays; the Boolean
mask is copied to the target backend.

# Arguments
* `pattern`      - integer indices or Boolean mask
* `shape::Tuple` - size of the array to sample (integer indices only)
* `S`            - storage type of the operator
"""
function LinearOperatorCollection.SamplingOp(::Type{T};
  pattern::P, shape::Tuple=(), S = Vector{T}) where {P} where T <: Number
  if length(shape) == 0
    return SamplingOpImpl(T, pattern; S = S)
  else
    return SamplingOpImpl(T, pattern, shape, S = S;)
  end
end


"""
 idx contains sampling index (for the first dimension) of a multidimensional Array
 of size "shape". Transform this into idx into the corresponding vector index
"""
function vectorizePattern(idx::T, shape::Tuple) where T<:AbstractArray{Int}
  return [ floor(Int,(i-1)/size(idx,1))*shape[1]+idx[i] for i = 1:length(idx) ]
end

struct SamplingOpImpl{T, vecT <: AbstractVector, O <: AbstractLinearOperator{T}, P <: AbstractArray, N} <: SamplingOp{T}
  op::O
  # effective pattern used by `op`: host integer indices or the owned Boolean mask
  pattern::P
  shape::NTuple{N, Int}
  function SamplingOpImpl(op::O, pattern::P, shape::NTuple{N, Int}) where {T, O <: AbstractLinearOperator{T}, P <: AbstractArray, N}
    return new{T, storage_type(op), O, P, N}(op, pattern, shape)
  end
end

function SamplingOpImpl(T::Type{<:Number}, pattern::AbstractArray{Int}, shape::Tuple; S = Vector{T})
  ndims(pattern)>1 ?  idx = vectorizePattern(pattern, shape) : idx = pattern
  op = opEye(T,length(idx), S = S)*opRestriction(idx, prod(shape); S = S)
  return SamplingOpImpl(op, idx, map(Int, shape))
end

function SamplingOpImpl(T::Type{<:Number}, pattern::AbstractArray{Bool}; S = Vector{T})
  pattern = copyto!(similar(S(undef,0), Bool, size(pattern)...), pattern)
  
  function prod!(res::AbstractArray{U}, x::AbstractArray{V}) where {U,V}
    res .= pattern.*x
  end

  op = LinearOperator(T, length(pattern), length(pattern), true, false, prod!; S = S)
  return SamplingOpImpl(op, pattern, ())
end

function Base.getproperty(op::SamplingOpImpl, field::Symbol)
  if in(field, (:op, :pattern, :shape))
    return getfield(op, field)
  else
    return getproperty(getfield(op, :op), field)
  end
end
Base.setproperty!(op::SamplingOpImpl, field::Symbol, value) = setproperty!(op.op, field, value)

storage_type(::SamplingOpImpl{T, vecT}) where {T, vecT} = vecT

# Rebuild from the effective pattern: integer indices are copied, the constructor copies the mask.
_rebuild_sampling(op::SamplingOpImpl{T, V, O, <:AbstractArray{Bool}}, S) where {T, V, O} =
  SamplingOp(T; pattern = op.pattern, S = S)
_rebuild_sampling(op::SamplingOpImpl{T}, S) where T =
  SamplingOp(T; pattern = copy(op.pattern), shape = op.shape, S = S)

Base.copy(op::SamplingOpImpl) = _rebuild_sampling(op, storage_type(op))

function Adapt.adapt_structure(to, op::SamplingOpImpl)
  target_storage = _adapt_target_storage(to, op.Mv, "SamplingOp")
  return _rebuild_sampling(op, typeof(target_storage))
end
