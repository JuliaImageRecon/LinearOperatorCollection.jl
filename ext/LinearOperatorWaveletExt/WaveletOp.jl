
"""
  WaveletOp(shape, wt=wavelet(WT.db2))

returns a `ẀaveletOp <: AbstractLinearOperator`, which performs a Wavelet transform on
a given input array.

# Arguments

* `shape`                 - size of the Array to transform
* (`wt=wavelet(WT.db2)`)  - Wavelet to apply
"""
function LinearOperatorCollection.WaveletOp(::Type{T}; shape::Tuple, wt=wavelet(WT.db2), S = Vector{T}) where T <: Number
  shape = filter(x-> x != 1, shape) # Drop dimension with 1
  tmp = Array{T}(undef, shape...)
  tmpRes = Array{T}(undef, shape...)
  return LinearOperator(T, prod(shape), prod(shape), false, false
            , (res,x)->prodwt!(res, x, wt, tmp, tmpRes)
            , nothing
            , (res,x)->ctprodwt!(res, x, wt, tmp, tmpRes), S = S)
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