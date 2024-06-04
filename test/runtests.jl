using LinearOperatorCollection
using Test
using Random
using LinearAlgebra
using FFTW
using Wavelets
using NFFT
using JLArrays

arrayTypes = [Array, JLArray]

@testset "LinearOperatorCollection" begin
  include("testNormalOp.jl")
  include("testOperators.jl")
end
