export FFTOpImpl

mutable struct FFTOpImpl{T, vecT, P <: AbstractFFTs.Plan{T}, IP <: AbstractFFTs.Plan{T}} <: FFTOp{T}
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
  const iplan :: IP
  const shift::Bool
  const unitary::Bool
  # The plan constructor kwargs are kept so Adapt can replay the same
  # planner configuration for a different array type.  `nothing` means that
  # the caller explicitly opted out with `cache=false`.
  const plan_kwargs::Union{Nothing,Dict{Symbol,Any}}
end

LinearOperators.storage_type(::FFTOpImpl{T, vecT}) where {T,vecT} = vecT

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
* (`cache=true`) - cache the FFT plan kwargs for `Adapt.adapt`; set to
  `false` to opt out of that reconstruction support

`Adapt.adapt(target, op)` preserves the shape, shift, and normalization
convention, planner options, and rebuilds the plans, workspaces, and closures
for `target`. If `cache=false`, adapting the operator raises an explicit
`ArgumentError` because the planner options were not retained.
"""
function LinearOperatorCollection.FFTOp(T::Type; shape::NTuple{D,Int64}, shift::Bool=true, unitary::Bool=true, S = Vector{Complex{real(T)}}, cache::Bool=true, kwargs...) where D
  
  tmpVec = similar(S(undef, 0), shape...)
  plan = plan_fft!(tmpVec; kwargs...)
  iplan = plan_bfft!(tmpVec; kwargs...)

  plan_kwargs = cache ? Dict{Symbol,Any}(pairs(kwargs)) : nothing

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
        0, 0, 0, similar(tmpVec, 0), similar(tmpVec, 0), plan, iplan, shift, unitary,
        plan_kwargs)
  end
end

function Adapt.adapt_structure(to, op::FFTOpImpl)
  target_storage = LinearOperatorCollection._adapt_target_storage(to, op.Mv, "FFTOp")
  op.plan_kwargs === nothing && throw(ArgumentError(
    "cannot adapt FFTOp constructed with cache=false: FFT plan kwargs were not retained"))
  return FFTOp(eltype(op); shape=size(op.plan), shift=op.shift,
               unitary=op.unitary, S=typeof(target_storage), cache=true,
               op.plan_kwargs...)
end

function fft_multiply!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, ::NTuple{D}, factor::T, tmpVec::AbstractArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
  plan * copyto!(tmpVec, x)
  res .= factor .* vec(tmpVec)
end

function fft_multiply_shift!(res::AbstractVector{T}, plan::P, x::AbstractVector{Tr}, shape::NTuple{D}, factor::T, tmpVec::AbstractArray{T,D}) where {T, Tr, P<:AbstractFFTs.Plan, D}
  ifftshift!(tmpVec, reshape(x,shape))
  plan * tmpVec
  fftshift!(reshape(res,shape), tmpVec)
  res .*= factor
end


function Base.copy(S::FFTOpImpl)
  return FFTOp(eltype(S); shape=size(S.plan), shift=S.shift, unitary=S.unitary,
               S = LinearOperators.storage_type(S), cache=!isnothing(S.plan_kwargs))
end
