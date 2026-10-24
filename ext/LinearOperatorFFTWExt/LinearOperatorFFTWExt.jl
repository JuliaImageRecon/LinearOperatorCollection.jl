module LinearOperatorFFTWExt

using LinearOperatorCollection, FFTW, FFTW.AbstractFFTs
using Adapt

include("FFTOp.jl")
include("DCTOp.jl")
include("DSTOp.jl")

end
