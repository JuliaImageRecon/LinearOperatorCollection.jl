module LinearOperatorCollection

using LinearAlgebra
using LinearAlgebra.BLAS: gemv, gemv!
using LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using SparseArrays
using Random
using InteractiveUtils

import Base: *, ∘, copy, getproperty, setproperty!
import LinearOperators: storage_type

using Reexport
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
       SamplingOp, NormalOp, WeightingOp, GradientOp, RadonOp

abstract type AbstractLinearOperatorFromCollection{T} <: AbstractLinearOperator{T} end
abstract type WaveletOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type FFTOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type DCTOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type DSTOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type NFFTOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type SamplingOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type NormalOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type GradientOp{T} <: AbstractLinearOperatorFromCollection{T} end
abstract type RadonOp{T} <: AbstractLinearOperatorFromCollection{T} end


"""
  returns a list of currently implemented `LinearOperator`s
"""
function linearOperatorList()
  return subtypes(AbstractLinearOperatorFromCollection)
end

# TODO (except for basetype) copied from RegLS, maybe place in utility package
function filterKwargs(T::Type, kwargWarning, kwargs)
  # If we don't take the basetype we won't find any methods
  baseType = Base.typename(T).wrapper # https://github.com/JuliaLang/julia/issues/35543
  table = methods(baseType)
  keywords = union(Base.kwarg_decl.(table)...)
  filtered = filter(in(keywords), keys(kwargs))

  if length(filtered) < length(kwargs) && kwargWarning
    filteredout = filter(!in(keywords), keys(kwargs))
    @warn "The following arguments were passed but filtered out: $(join(filteredout, ", ")). Please watch closely if this introduces unexpexted behaviour in your code."
  end

  return [key=>kwargs[key] for key in filtered]
end


# Next we create the factory methods. We probably want to given
# a better error message if the extension module is not loaded
for op in linearOperatorList()
  @eval begin
    function createLinearOperator(::Type{ Op }; kwargWarning::Bool = true, kwargs...) where Op <: $op{T} where T <: Number
      return $op(T; filterKwargs(Op,kwargWarning,kwargs)...)
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


include("WeightingOp.jl")
include("ProdOp.jl")
include("GradientOp.jl")
include("SamplingOp.jl")
include("NormalOp.jl")
include("DiagOp.jl")

end
