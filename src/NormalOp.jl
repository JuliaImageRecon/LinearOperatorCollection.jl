export NormalOp, normalOperator

mutable struct NormalOp{T,S,D,V} <: AbstractLinearOperator{T}
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
  Mv5 :: Vector{T}
  Mtu5 :: Vector{T}
  parent::S
  weights::D
  tmp::V
end

LinearOperators.storage_type(op::NormalOp) = typeof(op.Mv5)

function NormalOp(parent, weights)
  T = promote_type(eltype(parent), eltype(weights))
  tmp = Vector{T}(undef, size(parent, 1))
  return NormalOp(parent, weights, tmp)
end

function NormalOp(parent, weights, tmp::Vector{T}) where T

  function produ!(y, parent, tmp, x)
    mul!(tmp, parent, x)
    mul!(tmp, weights, tmp) # This can be dangerous. We might need to create two tmp vectors
    return mul!(y, adjoint(parent), tmp)
  end

  return NormalOp(size(parent,2), size(parent,2), false, false
         , (res,x) -> produ!(res, parent, tmp, x)
         , nothing
         , nothing
         , 0, 0, 0, false, false, false, T[], T[]
         , parent, weights, tmp)
end

function Base.copy(S::NormalOp)
  return NormalOp(copy(S.parent), S.weights, copy(S.tmp))
end

function normalOperator(parent, weights=opEye(eltype(parent), size(parent,1)))
  return NormalOp(parent, weights)
end

