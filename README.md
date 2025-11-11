# LinearOperatorCollection

[![Build Status](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![codecov](https://codecov.io/github/JuliaImageRecon/LinearOperatorCollection.jl/graph/badge.svg?token=MEjup4lqjO)](https://codecov.io/github/JuliaImageRecon/LinearOperatorCollection.jl)

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliaimagerecon.github.io/LinearOperatorCollection.jl/dev/)


## Purpose

This package contains a collection of linear operators that are in particular useful for multi-dimensional signal and image processing tasks. All operators are build using the LinearOperators.jl base type and derive from `AbstractLinearOperator`. For example this package
provides operators for the FFT (Fast Fourier Transform) and its non-equidistant variant (NFFT), the DCT (Discrete Cosine Transform), and the Wavelet transform. This package, however, does not implement
these transformation itself but uses established libraries for them. So in fact, LinearOperatorCollection's main purpose is to add a wrapper around low-level libraries like
FFTW.jl, NFFT.jl and NonuniformFFTs.jl, which allows to use the transformation as if they would be linear operators, i.e. implement `Op * x`, `adjoint(Op) * x` and the `mul!` based in-place variants of the former.

## Installation

Within Julia, use the package manager to install this package:
```julia
using Pkg
Pkg.add("LinearOperatorCollection")
```

## Usage
After loading the package one can construct an operator `Op` using the generic syntax
```julia
op = Op(T; kargs...) 
```
Here, `T` is the element type of the operator that usually should match the element type that
the operator will later operate on. The keyword arguments are operator specific but there are
 some common parameters. For instance, most operators have a `shape` parameter, which encodes
 the size of the vector `x`, the operator is applied to

### Extensions
To keep the load time of this package low, many operators are implemented using package extensions.
For instance, in order to get the `FFTOp`, one needs to load not only `LinearOperatorCollection` but
also `FFTW`:
```julia
using LinearOperatorCollection, FFTW
```
Small operators are implemented in LinearOperatorCollection directly.

### Example

The following shows how to build a two dimensional FFT operator and apply it to an image:
```julia
using LinearOperatorCollection, FFTW, LinearAlgebra

N = (64, 64)
x = vec( rand(ComplexF64, N) ) # The image needs to be vectorized so that the operator can be applied

F = FFTOp(ComplexF64, shape=N, shift=true) # shift will apply fftshifts before and after the FFT

# apply operator
y = F * x
# apply the adjoint operator
z = adjoint(F) * y

# apply the in place variants, which do not allocate memory during the computation
mul!(y, F, x)
mul!(z, adjoint(F), x)
```

### Implemented operators

Currently the following operators are implemented:
* `WeightingOp`: A diagonal weighting matrix
* `SamplingOp`: An operator which (sub)-samples the input vector
* `NormalOp`: An operator building the normal matrix `A' W A` in a lazy fashion
* `GradientOp`: An operator calculating the gradient along one or several directions
* `FFTOp`: An operator applying the fast Fourier transform
* `DCTOp`: An operator applying the discrete cosine transform
* `DSTOp`: An operator applying the discrete sine transform
* `NFFTOp`: An operator applying the non-equidistant fast Fourier transform
* `WaveletOp`: An operator applying the wavelet transformation

### Factory method

One can also create operators using a factory method that takes as input the abstract type. The above
operator can be also created by
```julia
F = createLinearOperator(FFTOp{ComplexF64}, shape=N, shift=true)
```
This is useful in cases where the operator should be exchangeable. A list of all implemented
can be obtained by calling
```julia
list = linearOperatorList()
```

