# LinearOperatorCollection

*Collection of linear operators for multi-dimensional signal and imaging tasks*


## Introduction

This package contains a collection of linear operators that are particularly useful for multi-dimensional signal and image processing tasks. Linear operators or linear maps behave like matrices in a matrix-vector product, but aren't necessarily matrices themselves. They can utilize more effective algorithms and can defer their computation until they are multiplied with a vector.

All operators provided by this package extend types and methods [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl). For example this package
provides operators for the FFT (Fast Fourier Transform) and its non-equidistant variant (NFFT), the DCT (Discrete Cosine Transform), and the Wavelet transform. This package, however, does not implement
these transformation itself but uses established libraries for them.

LinearOperatorCollection's main purpose is provide a wrapper around low-level libraries like FFTW.jl and NFFT.jl, which allows using the transformations as linear operators, i.e., implementing `Op * x`, `adjoint(Op) * x` and the `mul!` based in-place variants of the former.

## Installation

Within Julia, use the package manager to install this package:
```julia
using Pkg
Pkg.add("LinearOperatorCollection")
```
This will install `LinearOperatorCollection` and a subset of the available operators.
To keep the load time of this package low, many operators are implemented using package extensions.
For instance, in order to get the `FFTOp`, one needs to install not only `LinearOperatorCollection` but also `FFTW` and load both in a Julia sessiong:
```julia
Pkg.add("FFTW")
using LinearOperatorCollection, FFTW
```
Small operators are implemented in LinearOperatorCollection directly.


## License / Terms of Usage

The source code of this project is licensed under the MIT license. This implies that
you are free to use, share, and adapt it. However, please give appropriate credit
by citing the project.

## Contact

If you have problems using the software, find mistakes, or have general questions please use
the [issue tracker](hthttps://github.com/JuliaImageRecon/LinearOperatorCollection.jl/issues) to contact us.

## Related Packages

There exist many related packages which also implement efficient and/or lazy operators:

* [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl)
* [LinearMaps.jl](https://github.com/JuliaLinearAlgebra/LinearMaps.jl)
* [LazyArrays.jl](https://github.com/JuliaArrays/LazyArrays.jl)
* [BlockArrays.jl](https://github.com/JuliaArrays/BlockArrays.jl)
* [Kronecker.jl](https://github.com/MichielStock/Kronecker.jl)

Generally, it should be possible to combine operators and arrays from various packages.
