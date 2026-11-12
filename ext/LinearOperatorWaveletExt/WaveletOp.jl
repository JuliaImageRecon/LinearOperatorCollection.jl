
"""
  WaveletOp(T::Type; shape::Tuple, wt=wavelet(WT.db2), S = Vector{T})

returns a `WaveletOp <: AbstractLinearOperator`, which performs a Wavelet transform on
a given input array.

`Adapt.adapt(target, op)` and `copy(op)` rebuild the operator, with new temporary arrays,
from its shape and wavelet. The transform always runs on the CPU: with another storage type
than `Vector`, input and output are copied through host arrays. See
[Moving an Operator to Another Backend](@ref) for the supported targets.

# Arguments

* `shape`                 - size of the Array to transform
* (`wt=wavelet(WT.db2)`)  - Wavelet to apply
* (`S = Vector{T}`)       - storage type of the operator
"""
function LinearOperatorCollection.WaveletOp(::Type{T}; shape::Tuple, wt=wavelet(WT.db2), S = Vector{T}) where T <: Number
  shape = map(Int, filter(x-> x != 1, shape)) # Drop dimension with 1
  tmp = Array{T}(undef, shape...)
  tmpRes = Array{T}(undef, shape...)
  op = LinearOperator(T, prod(shape), prod(shape), false, false
            , (res,x)->prodwt!(res, x, wt, tmp, tmpRes)
            , nothing
            , (res,x)->ctprodwt!(res, x, wt, tmp, tmpRes), S = S)
  return WaveletOpImpl(op, shape, wt, tmp, tmpRes)
end

struct WaveletOpImpl{T, vecT <: AbstractVector, O <: AbstractLinearOperator{T}, N, W} <: WaveletOp{T}
  op::O
  # shape without the dimensions of size 1
  shape::NTuple{N, Int}
  wt::W
  # host arrays the transform reads from and writes to
  tmp::Array{T, N}
  tmpRes::Array{T, N}
  function WaveletOpImpl(op::O, shape::NTuple{N, Int}, wt::W, tmp::Array{T, N}, tmpRes::Array{T, N}) where {T, O <: AbstractLinearOperator{T}, N, W}
    return new{T, LinearOperators.storage_type(op), O, N, W}(op, shape, wt, tmp, tmpRes)
  end
end

function Base.getproperty(op::WaveletOpImpl, field::Symbol)
  if in(field, (:op, :shape, :wt, :tmp, :tmpRes))
    return getfield(op, field)
  else
    return getproperty(getfield(op, :op), field)
  end
end
Base.setproperty!(op::WaveletOpImpl, field::Symbol, value) = setproperty!(op.op, field, value)

LinearOperators.storage_type(::WaveletOpImpl{T, vecT}) where {T, vecT} = vecT

Base.copy(op::WaveletOpImpl) = WaveletOp(eltype(op); shape = op.shape, wt = op.wt, S = LinearOperators.storage_type(op))

function Adapt.adapt_structure(to, op::WaveletOpImpl)
  target_storage = LinearOperatorCollection._adapt_target_storage(to, op.Mv, "WaveletOp")
  return WaveletOp(eltype(op); shape = op.shape, wt = op.wt, S = typeof(target_storage))
end

prodwt!(res::Vector{T}, x::Vector{T}, wt, tmp::Array{T, D}, tmpRes::Array{T, D}) where {T, D} = dwt!(reshape(res,size(tmpRes)), reshape(x,size(tmpRes)), wt)
function prodwt!(res::vecT, x::vecT, wt, tmp::Array{T, D}, tmpRes::Array{T, D}) where {T, D, vecT <: AbstractArray{T}}
  copyto!(tmp, x)
  dwt!(tmpRes, tmp, wt)
  copyto!(res, tmpRes)
end

ctprodwt!(res::Vector{T}, x::Vector{T}, wt, tmp::Array{T, D}, tmpRes::Array{T, D}) where {T, D} = idwt!(reshape(res,size(tmpRes)), reshape(x,size(tmpRes)), wt)
function ctprodwt!(res::vecT, x::vecT, wt, tmp::Array{T, D}, tmpRes::Array{T, D}) where {T, D, vecT <: AbstractArray{T}}
  copyto!(tmp, x)
  idwt!(tmpRes, tmp, wt)
  copyto!(res, tmpRes)
end