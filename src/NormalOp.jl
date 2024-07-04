export normalOperator

function LinearOperatorCollection.NormalOp(::Type{T}; parent, weights = opEye(eltype(parent), size(parent, 1), S = storage_type(parent))) where T <: Number
  return NormalOp(T, parent, weights)
end

# TODO Are weights always restricted to T or can they also be real(T)?
function NormalOp(::Type{T}, parent, ::Nothing) where T
  weights = opEye(eltype(parent), size(parent, 1), S = storage_type(parent))
  return NormalOp(T, parent, weights)
end
NormalOp(::Type{T}, parent, weights::AbstractVector{T}) where T = NormalOp(T, parent, WeightingOp(weights))

NormalOp(::Type{T}, parent, weights::AbstractLinearOperator{T}; kwargs...) where T = NormalOpImpl(parent, weights)

mutable struct NormalOpImpl{T,S,D,V} <: NormalOp{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Nothing
  ctprod! :: Nothing
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  Mv5 :: V
  Mtu5 :: V
  parent::S
  weights::D
  tmp::V
end

LinearOperators.storage_type(op::NormalOpImpl) = typeof(op.Mv5)

function NormalOpImpl(parent, weights)
  S = promote_type(storage_type(parent), storage_type(weights))
  isconcretetype(S) || throw(LinearOperatorException("Storage types cannot be promoted to a concrete type"))
  tmp = S(undef, size(parent, 1))
  return NormalOpImpl(parent, weights, tmp)
end

function NormalOpImpl(parent, weights, tmp)
  function produ!(y, parent, weights, tmp, x)
    mul!(tmp, parent, x)
    mul!(tmp, weights, tmp) # This can be dangerous. We might need to create two tmp vectors
    mul!(tmp, weights, tmp)
    return mul!(y, adjoint(parent), tmp)
  end

  return NormalOpImpl{eltype(parent), typeof(parent), typeof(weights), typeof(tmp)}(size(parent,2), size(parent,2), false, false
         , (res,x) -> produ!(res, parent, weights, tmp, x)
         , nothing
         , nothing
         , 0, 0, 0, true, false, true, similar(tmp, 0), similar(tmp, 0)
         , parent, weights, tmp)
end

function Base.copy(S::NormalOpImpl)
  return NormalOpImpl(copy(S.parent), S.weights, copy(S.tmp))
end

function normalOperator(parent, weights=opEye(eltype(parent), size(parent, 1), S= storage_type(parent)); kwargs...)
  return NormalOp(eltype(storage_type((parent))); parent = parent, weights = weights)
end