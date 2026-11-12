using Metal
using LinearOperatorCollection
using LinearOperatorCollection.OhMyThreads
using Adapt
using JLArrays
using Test
using Random
using LinearAlgebra
using FFTW
using Wavelets
using NonuniformFFTs
using NFFT
using RadonKA

# Metal has no Float64, so runtests.jl cannot run here. This runs the Float32 and ComplexF32
# nested operators, including SamplingOp ∘ FFTOp, and a Float32 WaveletOp moved to MtlArray
# and compared with the CPU.

# Load the test helpers without the testsets of the files
include(ex -> Meta.isexpr(ex, :macrocall) ? nothing : ex, joinpath(@__DIR__(), "..", "testAdapt.jl"))
include(ex -> Meta.isexpr(ex, :macrocall) ? nothing : ex, joinpath(@__DIR__(), "..", "testOperators.jl"))

@testset "LinearOperatorCollection on Metal" begin
  N = 16
  @testset "ProdOp, NormalOpImpl, and ProdNormalOp recurse" check_nested_prod_adapt(MtlArray, N)
  @testset "SamplingOp ∘ FFTOp" check_sampling_fft_adapt(MtlArray, N)
  @testset "WaveletOp" testWaveletAdapt(64, 60, Float32; arrayType = MtlArray)
end
