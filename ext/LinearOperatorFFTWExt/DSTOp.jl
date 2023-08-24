export DSTOpImpl

function LinearOperatorCollection.constructLinearOperator(::Type{Op};
  shape::Tuple, shift::Bool=true, unitary::Bool=true, cuda::Bool=false) where Op <: DSTOp{T} where T <: Number
  return DSTOpImpl(T, shape)
end

mutable struct DSTOpImpl{T} <: DSTOp{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Nothing
  ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  Mv5 :: Vector{T}
  Mtu5 :: Vector{T}
  plan
  iplan
end

LinearOperators.storage_type(op::DSTOpImpl) = typeof(op.Mv5)

"""
  DSTOpImpl(T::Type, shape::Tuple)

returns a `LinearOperator` which performs a DST on a given input array.

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
"""
function DSTOpImpl(T::Type, shape::Tuple)
  tmp=Array{Complex{real(T)}}(undef, shape)

  plan = FFTW.plan_r2r!(tmp,FFTW.RODFT10)
  iplan = FFTW.plan_r2r!(tmp,FFTW.RODFT01)

  w = weights(shape, T)

  return DSTOpImpl{T}(prod(shape), prod(shape), true, false
            , (res,x) -> dst_multiply!(res,plan,x,tmp,w)
            , nothing
            , (res,x) -> dst_bmultiply!(res,iplan,x,tmp,w)
            , 0, 0, 0, true, false, true, T[], T[]
            , plan
            , iplan)
end

function weights(s, T::Type)
  w = ones(T,s...)./T(sqrt(8*prod(s)))
  w[s[1],:,:]./= T(sqrt(2))
  if length(s)>1
    w[:,s[2],:]./= T(sqrt(2))
    if length(s)>2
      w[:,:,s[3]]./= T(sqrt(2))
    end
  end
  return reshape(w,prod(s))
end

function dst_multiply!(res::Vector{T}, plan::P, x::Vector{T}, tmp::Array{T,D}, weights::Vector{T}) where {T,P,D}
  tmp[:] .= x
  plan * tmp
  res .= vec(tmp).*weights
end

function dst_bmultiply!(res::Vector{T}, plan::P, x::Vector{T}, tmp::Array{T,D}, weights::Vector{T}) where {T,P,D}
  tmp[:] .= x./weights
  plan * tmp
  res[:] .= vec(tmp)./(8*length(tmp))
end

function Base.copy(S::DSTOpImpl)
  return DSTOpImpl(eltype(S), size(S.plan))
end