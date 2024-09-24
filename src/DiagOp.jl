export DiagOp

mutable struct DiagOp{T, vecT, vecO} <: AbstractLinearOperator{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Function
  ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  Mv5 :: vecT
  Mtu5 :: vecT
  ops :: vecO
  equalOps :: Bool
  xIdx :: Vector{Int64}
  yIdx :: Vector{Int64}
end


LinearOperators.storage_type(op::DiagOp) = typeof(op.Mv5)



"""
    DiagOp(ops...)
    DiagOp(ops::Vector{...})
    DiagOp(ops::NTuple{N,...})

create a bloc-diagonal operator out of the `LinearOperator`s  or `Array`s contained in ops
"""
function DiagOp(ops)
  nrow = 0
  ncol = 0
  S = LinearOperators.storage_type(first(ops))
  for i = 1:length(ops)
    nrow += size(ops[i], 1)
    ncol += size(ops[i], 2)
    S = promote_type(S, LinearOperators.storage_type(ops[i]))
  end
  isconcretetype(S) || throw(LinearOperatorException("Storage types cannot be promoted to a concrete type"))

  xIdx = cumsum(vcat(1,[size(ops[i], 2) for i=1:length(ops)]))
  yIdx = cumsum(vcat(1,[size(ops[i], 1) for i=1:length(ops)]))

  Op = DiagOp{eltype(first(ops)), S, typeof(ops)}( nrow, ncol, false, false,
                     (res,x) -> (diagOpProd(res,x,nrow,xIdx,yIdx,ops)),
                     (res,y) -> (diagOpTProd(res,y,ncol,yIdx,xIdx,ops)),
                     (res,y) -> (diagOpCTProd(res,y,ncol,yIdx,xIdx,ops)),
                     0, 0, 0, false, false, false, S(undef, 0), S(undef, 0),
                     ops, false, xIdx, yIdx)

  return Op
end
DiagOp(ops...) = DiagOp(collect(ops))

function DiagOp(op::Union{AbstractLinearOperator{T}, AbstractArray{T}}, N::Int64=1; copyOpsFn = copy) where T <: Number
  ops = [copyOpsFn(op) for n=1:N]
  op = DiagOp(ops)
  op.equalOps = true
  return op
end

function diagOpProd(y::AbstractVector{T}, x::AbstractVector{T}, nrow::Int, xIdx, yIdx, ops) where T
  for i=1:length(ops)
    mul!(view(y,yIdx[i]:yIdx[i+1]-1), ops[i], view(x,xIdx[i]:xIdx[i+1]-1))
  end
  return y
end

function diagOpTProd(y::AbstractVector{T}, x::AbstractVector{T}, ncol::Int, xIdx, yIdx, ops) where T
  for i=1:length(ops)
    mul!(view(y,yIdx[i]:yIdx[i+1]-1), transpose(ops[i]), view(x,xIdx[i]:xIdx[i+1]-1))
  end
  return y
end

function diagOpCTProd(y::AbstractVector{T}, x::AbstractVector{T}, ncol::Int, xIdx, yIdx, ops) where T
  for i=1:length(ops)
    mul!(view(y,yIdx[i]:yIdx[i+1]-1), adjoint(ops[i]), view(x,xIdx[i]:xIdx[i+1]-1))
  end
  return y
end

### Normal Matrix Code ###

mutable struct DiagNormalOp{T,vecT,V} <: AbstractLinearOperator{T}
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
  Mv5 :: vecT
  Mtu5 :: vecT
  normalOps::V
  idx::Vector{Int64}
  y::vecT
end

LinearOperators.storage_type(op::DiagNormalOp) = typeof(op.Mv5)

function DiagNormalOp(normalOps, N, idx, y::AbstractVector{T}) where {T}

  S = LinearOperators.storage_type(first(normalOps))
  for nop in normalOps
    S = promote_type(S, LinearOperators.storage_type(nop))
  end

  return DiagNormalOp{eltype(first(normalOps)), S, typeof(normalOps)}(N, N, false, false
         , (res,x) -> diagNormOpProd!(res, normalOps, idx, x)
         , nothing
         , nothing
         , 0, 0, 0, false, false, false, S(undef, 0), S(undef, 0)
         , normalOps, idx, y)
end

function diagNormOpProd!(y, normalOps, idx, x)
  for i=1:length(normalOps)
    mul!(view(y,idx[i]:idx[i+1]-1), normalOps[i], view(x,idx[i]:idx[i+1]-1))
 end
 return y
end

function LinearOperatorCollection.normalOperator(diag::DiagOp, W=opEye(eltype(diag), size(diag,1), S = LinearOperators.storage_type(diag)); copyOpsFn = copy, kwargs...)
  T = promote_type(eltype(diag), eltype(W))
  S = promote_type(LinearOperators.storage_type(diag), LinearOperators.storage_type(W))
  isconcretetype(S) || throw(LinearOperatorException("Storage types cannot be promoted to a concrete type"))
  tmp = S(undef, diag.nrow)
  tmp .= one(eltype(diag))
  weights = W*tmp


  if diag.equalOps
    # this optimization is only allowed if all ops are the same

    # we promote the weights to be of the same type as T, which will be required
    # when creating the temporary vector in normalOperator in a later stage
    op = DiagNormalOp([normalOperator(copyOpsFn(diag.ops[1]), WeightingOp(T; weights=T.(weights[diag.yIdx[i]:diag.yIdx[i+1]-1])); copyOpsFn = copyOpsFn, kwargs...)
                     for i in 1:length(diag.ops)], size(diag,2), diag.xIdx, S(undef, diag.ncol) )
  else
    op = DiagNormalOp([normalOperator(diag.ops[i], WeightingOp(T; weights=T.(weights[diag.yIdx[i]:diag.yIdx[i+1]-1])); copyOpsFn = copyOpsFn, kwargs...)
                     for i in 1:length(diag.ops)], size(diag,2), diag.xIdx, S(undef, diag.ncol) )
  end

  return op
end