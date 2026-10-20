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
#using JLArrays

areTypesDefined = @isdefined arrayTypes
arrayTypes = areTypesDefined ? arrayTypes : [Array] #, JLArray]

@testset "LinearOperatorCollection" begin
  #include("testAqua.jl")
  include("testAdapt.jl")
  include("testNormalOp.jl")
  include("testOperators.jl")
end
