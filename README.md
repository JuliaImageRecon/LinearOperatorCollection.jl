# LinearOperatorCollection

[![Build Status](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl/actions/workflows/CI.yml?query=branch%3Amain)

[![codecov.io](http://codecov.io/JuliaImageRecon/LinearOperatorCollection.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaImageRecon/LinearOperatorCollection.jl?branch=master)

## Purpose

This package contains a collection of linear operators that are in particular useful for multi-dimensional signal and image processing tasks. All operators are build using the LinearOperators.jl base type and derive from `AbstractLinearOperator`. For example this package
provides operators for the FFT (Fast Fourier Transform) and its non-equidistant variant (NFFT), the DCT (Discrete Cosine Transform), and the Wavelet transform. This package, however, does not implement
these transformation itself but uses established libraries for them. So in fact, LinearOperatorCollection's main purpose is to add a wrapper around low-level libraries like
FFTW.jl and NFFT.jl, which allows to use the transformation as if they would be linear operators, i.e. implement `Op * x`, `adjoint(Op) * x` and the `mul!` based in-place variants of the former.

## Installation

Within Julia, use the package manager to install this package:
```julia
using Pkg
Pkg.add("LinearOperatorCollection")
```

## Usage
...