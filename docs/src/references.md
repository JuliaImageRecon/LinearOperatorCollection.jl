```@docs
SamplingOp
WeightingOp
DiagOp
GradientOp
ProdOp
NormalOp
normalOperator
WaveletOp
RadonOp
NFFTOp
FFTOp
DCTOp
DSTOp
```



Modules = [LinearOperatorCollection,
isdefined(Base, :get_extension) ? Base.get_extension(LinearOperatorCollection, :LinearOperatorFFTWExt) : LinearOperatorCollection.LinearOperatorFFTWExt,
isdefined(Base, :get_extension) ? Base.get_extension(LinearOperatorCollection, :LinearOperatorNFFTWExt) : LinearOperatorCollection.LinearOperatorNFFTWExt,
isdefined(Base, :get_extension) ? Base.get_extension(LinearOperatorCollection, :LinearOperatoRadonWExt) : LinearOperatorCollection.LinearOperatorRadonExt,
isdefined(Base, :get_extension) ? Base.get_extension(LinearOperatorCollection, :LinearOperatorWaveletExt) : LinearOperatorCollection.LinearOperatorWaveletExt,
]
