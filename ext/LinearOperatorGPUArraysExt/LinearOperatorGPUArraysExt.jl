module LinearOperatorGPUArraysExt

using LinearOperatorCollection, GPUArrays, GPUArrays.KernelAbstractions # Hacky but with [KernelAbstractions, GPUArrays] the extension didnt trigger

include("GradientOp.jl")


end # module
