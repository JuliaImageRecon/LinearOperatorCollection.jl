export ProdOp
"""
    `mutable struct ProdOp{T}`

  struct describing the result of a composition/product of operators.
  Describing the composition using a dedicated type has the advantage
  that the latter can be made copyable. This is particularly relevant for
  multi-threaded code
"""
mutable struct ProdOp{T,U,V, vecT <: AbstractVector{T}} <: AbstractLinearOperatorFromCollection{T}
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
  A::U
  B::V
  tmp::vecT
end

"""
    ProdOp(A,B)

composition/product of two Operators. Differs with * since it can handle normal operator
"""
function ProdOp(A, B)
  nrow = size(A, 1)
  ncol = size(B, 2)
  S = promote_type(storage_type(A), storage_type(B))
  tmp_ = S(undef, size(B, 1))

  function produ!(res, x::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
    mul!(tmp, B, x)
    return mul!(res, A, tmp)
  end

  function tprodu!(res, y::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
    mul!(tmp, transpose(A), y)
    return mul!(res, transpose(B), tmp)
  end

  function ctprodu!(res, y::AbstractVector{T}, tmp) where T<:Union{Real,Complex}
    mul!(tmp, adjoint(A), y)
    return mul!(res, adjoint(B), tmp)
  end

  Op = ProdOp( nrow, ncol, false, false,
                     (res,x) -> produ!(res,x,tmp_),
                     (res,y) -> tprodu!(res,y,tmp_),
                     (res,y) -> ctprodu!(res,y,tmp_), 
                     0, 0, 0, false, false, false, similar(tmp_, 0), similar(tmp_, 0),
                     A, B, tmp_)

  return Op
end

function Base.copy(S::ProdOp{T}) where T
  A = copy(S.A)
  B = copy(S.B)
  return ProdOp(A,B)
end

Base.:*(::Type{<:ProdOp}, A, B) = ProdOp(A, B)
Base.:*(::Type{<:ProdOp}, A, args...) = ProdOp(A, *(ProdOp, args...))
Base.:∘(A::AbstractLinearOperator, B::AbstractLinearOperator) = ProdOp(A, B)

storage_type(op::ProdOp) = typeof(op.Mv5)

mutable struct ProdNormalOp{T,S,U,V <: AbstractVector{T}} <: AbstractLinearOperator{T}
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
  opOuter::S
  normalOpInner::U
  tmp::V
end

storage_type(op::ProdNormalOp) = typeof(op.Mv5)


function ProdNormalOp(opOuter, normalOpInner, tmp)

  function produ!(y, opOuter, normalOpInner, tmp, x)
    mul!(tmp, opOuter, x)
    mul!(tmp, normalOpInner, tmp) # This can be dangerous. We might need to create two tmp vectors
    return mul!(y, adjoint(opOuter), tmp)
  end

  return ProdNormalOp(size(opOuter,2), size(opOuter,2), false, false
         , (res,x) -> produ!(res, opOuter, normalOpInner, tmp, x)
         , nothing
         , nothing
         , 0, 0, 0, false, false, false, similar(tmp, 0), similar(tmp, 0)
         , opOuter, normalOpInner, tmp)
end


# In this case we are converting the left argument into a 
# weighting matrix, that is passed to normalOperator
# TODO Port vom MRIOperators drops given weighting matrix, I just left it out for now
normalOperator(S::ProdOp{T, <:WeightingOp, matT}) where matT = normalOperator(S.B, S.A)
function normalOperator(S::ProdOp, W=opEye(eltype(S),size(S,1), S = storage_type(S)))
  arrayType = storage_type(S)
  tmp = arrayType(undef, size(S.A, 2))
  return ProdNormalOp(S.B, normalOperator(S.A, W), tmp)
end

function Base.copy(S::ProdNormalOp) 
  opOuter = copy(S.opOuter)
  opInner = copy(S.normalOpInner)
  tmp = copy(S.tmp)
  return ProdNormalOp(opOuter, opInner, tmp)
end
