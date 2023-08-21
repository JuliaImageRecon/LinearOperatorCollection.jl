export WaveletOp

"""
  WaveletOp(shape, wt=wavelet(WT.db2))

returns a `ẀaveletOp <: AbstractLinearOperator`, which performs a Wavelet transform on
a given input array.

# Arguments

* `shape`                 - size of the Array to transform
* (`wt=wavelet(WT.db2)`)  - Wavelet to apply
"""
function WaveletOp(T::Type, shape, wt=wavelet(WT.db2))
  return LinearOperator(T, prod(shape), prod(shape), false, false
            , (res,x)->dwt!(reshape(res,shape), reshape(x,shape), wt)
            , nothing
            , (res,x)->idwt!(reshape(res,shape), reshape(x,shape), wt) )
end
