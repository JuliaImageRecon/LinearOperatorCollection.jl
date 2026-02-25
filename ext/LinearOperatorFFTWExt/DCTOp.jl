export DCTOpImpl

mutable struct DCTOpImpl{T, vecT, P} <: DCTOp{T}
  const nrow :: Int
  const ncol :: Int
  const symmetric :: Bool
  const hermitian :: Bool
  const prod! :: Function
  const tprod! :: Nothing
  const ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  Mv :: vecT
  Mtu :: vecT
  const plan :: P
  const dcttype::Int
end

LinearOperators.storage_type(::DCTOpImpl{T, vecT}) where {T,vecT} = vecT

"""
  DCTOpImpl(T::Type, shape::Tuple, dcttype=2)

returns a `DCTOpImpl <: AbstractLinearOperator` which performs a DCT on a given input array.

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* `dcttype`       - type of DCT (currently `2` and `4` are supported)
"""
function LinearOperatorCollection.DCTOp(T::Type; shape::Tuple, S = Array{T}, dcttype=2)

  tmp=similar(S(undef, 0), shape...)
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

  return DCTOpImpl{T, S, typeof(plan)}(prod(shape), prod(shape), false, false,
                      prod!, nothing, tprod!,
                      0, 0, 0, S(undef, 0), S(undef, 0),
                      plan, dcttype)
end

function dct_multiply2(res::AbstractVector{T}, plan::P, x::AbstractVector{T}, tmp::AbstractArray{T,D}) where {T,P,D}
  tmp[:] .= x
  plan * tmp
  res .= vec(tmp)
end

function dct_multiply4(res::AbstractVector{T}, plan::P, x::AbstractVector{T}, tmp::AbstractArray{T,D}, factor::T) where {T,P,D}
  tmp[:] .= x
  plan * tmp
  res .= factor.*vec(tmp)
end

function Base.copy(S::DCTOpImpl)
  return DCTOpImpl(eltype(S), size(S.plan), S.dcttype)
end