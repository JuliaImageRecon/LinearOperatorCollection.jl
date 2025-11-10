using LinearOperatorCollection
using Test
using Random
using LinearAlgebra
using FFTW
using Wavelets
using NonuniformFFTs
using NFFT
using RadonKA
using JLArrays

areTypesDefined = @isdefined arrayTypes
arrayTypes = areTypesDefined ? arrayTypes : [Array] #, JLArray]

@testset "LinearOperatorCollection" begin
  include("testNormalOp.jl")
  include("testOperators.jl")
end
