export DCTOpImpl

function LinearOperatorCollection.constructLinearOperator(::Type{Op};
  shape::Tuple, dcttype::Int) where Op <: DCTOp{T} where T <: Number
  return DCTOpImpl(T, shape, dcttype)
end

mutable struct DCTOpImpl{T} <: AbstractLinearOperatorFromCollection{T}
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
  dcttype::Int
end

LinearOperators.storage_type(op::DCTOpImpl) = typeof(op.Mv5)

"""
  DCTOpImpl(T::Type, shape::Tuple, dcttype=2)

returns a `DCTOpImpl <: AbstractLinearOperator` which performs a DCT on a given input array.

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* `dcttype`       - type of DCT (currently `2` and `4` are supported)
"""
function DCTOpImpl(T::Type, shape::Tuple, dcttype=2)

  tmp=Array{Complex{real(T)}}(undef, shape) 
  if dcttype == 2
    plan = plan_dct!(tmp)
    iplan = plan_idct!(tmp)
    prod! = (res, x)  -> dct_multiply2(res, plan, x, tmp)
    tprod! = (res, x)  -> dct_multiply2(res, iplan, x, tmp)

  elseif dcttype == 4
    factor = T(sqrt(1.0/(prod(shape)* 2^length(shape)) ))
    plan = FFTW.plan_r2r!(tmp,FFTW.REDFT11)
    prod! = (res, x) -> dct_multiply4(res, plan, x, tmp, factor)
    tprod! = (res, x) -> dct_multiply4(res, plan, x, tmp, factor)
  else
    error("DCT type $(dcttype) not supported")
  end

  return DCTOpImpl{T}(prod(shape), prod(shape), false, false,
                      prod!, nothing, tprod!,
                      0, 0, 0, true, false, true, T[], T[],
                      plan, dcttype)
end

function dct_multiply2(res::Vector{T}, plan::P, x::Vector{T}, tmp::Array{T,D}) where {T,P,D}
  tmp[:] .= x
  plan * tmp
  res .= vec(tmp)
end

function dct_multiply4(res::Vector{T}, plan::P, x::Vector{T}, tmp::Array{T,D}, factor::T) where {T,P,D}
  tmp[:] .= x
  plan * tmp
  res .= factor.*vec(tmp)
end

function Base.copy(S::DCTOpImpl)
  return DCTOpImpl(eltype(S), size(S.plan), S.dcttype)
end