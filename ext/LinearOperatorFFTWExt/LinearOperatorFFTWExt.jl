module LinearOperatorFFTWExt

using LinearOperatorCollection, FFTW, FFTW.AbstractFFTs

include("FFTOp.jl")
include("DCTOp.jl")
include("DSTOp.jl")

end