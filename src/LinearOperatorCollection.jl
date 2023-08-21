module LinearOperatorCollection

import Base: length, iterate, \
using LinearAlgebra
import LinearAlgebra.BLAS: gemv, gemv!
import LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using SparseArrays
using Random
#using CUDA

using Reexport
@reexport using Reexport
@reexport using LinearOperators
@reexport using FFTW
@reexport using Wavelets

LinearOperators.use_prod5!(op::opEye) = false
LinearOperators.has_args5(op::opEye) = false


const Trafo = Union{AbstractMatrix, AbstractLinearOperator, Nothing}
const FuncOrNothing = Union{Function, Nothing}

# Helper function to wrap a prod into a 5-args mul
function wrapProd(prod::Function)
  λ = (res, x, α, β) -> begin
    if β == zero(β)
      res .= prod(x) .* α
    else
      res .= prod(x) .* α .+ β .* res
    end
  end
  return λ
end

include("FFTOp.jl")
include("DCTOp.jl")
include("DSTOp.jl")
include("WaveletOp.jl")
include("GradientOp.jl")
include("SamplingOp.jl")
include("WeightingOp.jl")
include("NormalOp.jl")

export linearOperator, linearOperatorList

linearOperator(op::Nothing,shape,T::Type=ComplexF32) = nothing

"""
  returns a list of currently implemented `LinearOperator`s
"""
function linearOperatorList()
  return ["DCT-II", "DCT-IV", "FFT", "DST", "Wavelet", "Gradient"]
end

"""
    linearOperator(op::AbstractString, shape)

returns the `LinearOperator` with name `op`.

# valid names
* `"FFT"`
* `"DCT-II"`
* `"DCT-IV"`
* `"DST"`
* `"Wavelet"`
* `"Gradient"`
"""
function linearOperator(op::AbstractString, shape, T::Type=ComplexF32)
  shape_ = tuple(shape...)
  if op == "FFT"
    trafo = FFTOp(T, shape_, false) #FFTOperator(shape)
  elseif op == "DCT-II"
    shape_ = tuple(shape[shape .!= 1]...)
    trafo = DCTOp(T, shape_, 2)
  elseif op == "DCT-IV"
    shape_ = tuple(shape[shape .!= 1]...)
    trafo = DCTOp(T, shape_, 4)
  elseif op == "DST"
    trafo = DSTOp(T, shape_)
  elseif op == "Wavelet"
    trafo = WaveletOp(T,shape_)
  elseif op=="Gradient"
    trafo = GradientOp(T,shape_)
  else
    error("Unknown transformation")
  end
  trafo
end

end
