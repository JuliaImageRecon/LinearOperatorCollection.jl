export FFTOpImpl

mutable struct FFTOpImpl{T, vecT, P <: AbstractFFTs.Plan{T}, IP <: AbstractFFTs.Plan{T}} <: FFTOp{T}
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
  Mv5 :: vecT
  Mtu5 :: vecT
  plan :: P
  iplan :: IP
  shift::Bool
  unitary::Bool
end

LinearOperators.storage_type(op::FFTOpImpl) = typeof(op.Mv5)

"""
  FFTOp(T::Type; shape::Tuple, shift=true, unitary=true)

returns an operator which performs an FFT on Arrays of type T

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* (`shift=true`)  - if true, fftshifts are performed
* (`unitary=true`)  - if true, FFT is normalized such that it is unitary
* (`S = Vector{T}`) - type of temporary vector, change to use on GPU
* (`kwargs...`) - keyword arguments given to fft plan
"""
function LinearOperatorCollection.FFTOp(T::Type; shape::NTuple{D,Int64}, shift::Bool=true, unitary::Bool=true, S = Array{Complex{real(T)}}, kwargs...) where D
  
  tmpVec = S(undef, shape...)
  plan = plan_fft!(tmpVec; kwargs...)
  iplan = plan_bfft!(tmpVec; kwargs...)

  if unitary
    facF = T(1.0/sqrt(prod(shape)))
    facB = T(1.0/sqrt(prod(shape)))
  else
    facF = T(1.0)
    facB = T(1.0)
  end

  let shape_ = shape, plan_ = plan, iplan_ = iplan, tmpVec_ = tmpVec, facF_ = facF, facB_ = facB

    fun! = fft_multiply!
    if shift
      fun! = fft_multiply_shift!
    end

    return FFTOpImpl(prod(shape), prod(shape), false, false, (res, x) -> fun!(res, plan_, x, shape_, facF_, tmpVec_),
        nothing, (res, x) -> fun!(res, iplan_, x, shape_, facB_, tmpVec_),
        0, 0, 0, true, false, true, similar(tmpVec, 0), similar(tmpVec, 0), plan, iplan, shift, unitary)
  end
end

function fft_multiply!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, factor::T, tmpVec::AbstractArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
  tmpVec[:] .= x
  plan * tmpVec
  res .= factor .* vec(tmpVec)
end

function fft_multiply_shift!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, shape::NTuple{D}, factor::T, tmpVec::AbstractArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
  ifftshift!(tmpVec, reshape(x,shape))
  plan * tmpVec
  fftshift!(reshape(res,shape), tmpVec)
  res .*= factor
end


function Base.copy(S::FFTOpImpl)
  return FFTOp(eltype(S); shape=size(S.plan), shift=S.shift, unitary=S.unitary, S = LinearOperators.storage_type(S)) # TODO loses kwargs...
end
