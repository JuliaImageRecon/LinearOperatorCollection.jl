module LinearOperatorCollection

import Base: length, iterate, \
using LinearAlgebra
import LinearAlgebra.BLAS: gemv, gemv!
import LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using SparseArrays
using Random
using InteractiveUtils

using Reexport
@reexport using Reexport
@reexport using LinearOperators

LinearOperators.use_prod5!(op::opEye) = false
LinearOperators.has_args5(op::opEye) = false

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

export linearOperatorList, createLinearOperator
export AbstractLinearOperatorFromCollection, WaveletOp, FFTOp, DCTOp, DSTOp, NFFTOp,
       SamplingOp, NormalOp, WeightingOp, GradientOp

abstract type AbstractLinearOperatorFromCollection{T} <: AbstractLinearOperator{T} end
abstract type WaveletOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type FFTOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type DCTOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type DSTOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type NFFTOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type SamplingOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type NormalOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type WeightingOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type GradientOp{T} <: AbstractLinearOperatorFromCollection{T} end


"""
  returns a list of currently implemented `LinearOperator`s
"""
function linearOperatorList()
  return subtypes(AbstractLinearOperatorFromCollection)
end

# Next we create the factory methods. We probably want to given
# a better error message if the extension module is not loaded
for op in linearOperatorList()
  @eval begin
    function createLinearOperator(::Type{ Op }; kargs...) where Op <: $op{T} where T <: Number
      return $op(T; kargs...)
    end
  end
end

# String constructor
function createLinearOperator(op::String, ::Type{T}; kargs...) where T <: Number
  if contains(op, "DCT") 
    strToOp = Dict("DCT-I"=>(DCTOp{T},1), "DCT-II"=>(DCTOp{T},2), 
                              "DCT-III"=>(DCTOp{T},3), "DCT-IV"=>(DCTOp{T},4))
    trafo, dcttype = strToOp[op]  
    return createLinearOperator(trafo; dcttype, kargs...)
  elseif contains(op, "DST") 
    return createLinearOperator(DSTOp{T}; kargs...)
  elseif contains(op, "FFT") 
    return createLinearOperator(FFTOp{T}; kargs...)
  elseif contains(op, "NFFT") 
    return createLinearOperator(NFFTOp{T}; kargs...)
  elseif contains(op, "NFFT") 
    return createLinearOperator(NFFTOp{T}; kargs...)
  elseif contains(op, "Wavelet") 
    return createLinearOperator(WaveletOp{T}; kargs...)
  else
    error("Linear operator $(op) currently not implemented")
  end
end



include("GradientOp.jl")
include("SamplingOp.jl")
include("WeightingOp.jl")
include("NormalOp.jl")

end
