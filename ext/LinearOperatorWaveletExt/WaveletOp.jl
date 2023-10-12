
"""
  WaveletOp(shape, wt=wavelet(WT.db2))

returns a `ẀaveletOp <: AbstractLinearOperator`, which performs a Wavelet transform on
a given input array.

# Arguments

* `shape`                 - size of the Array to transform
* (`wt=wavelet(WT.db2)`)  - Wavelet to apply
"""
function LinearOperatorCollection.WaveletOp(::Type{T}; shape::Tuple, wt=wavelet(WT.db2)) where T <: Number
  shape = filter(x-> x != 1, shape) # Drop dimension with 1
  return LinearOperator(T, prod(shape), prod(shape), false, false
            , (res,x)->dwt!(reshape(res,shape), reshape(x,shape), wt)
            , nothing
            , (res,x)->idwt!(reshape(res,shape), reshape(x,shape), wt) )
end
