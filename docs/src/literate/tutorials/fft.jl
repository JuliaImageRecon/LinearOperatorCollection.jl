# # Fourier Operator
include("../../util.jl") #hide
# The Fourier operator and its related operators for the discrete cosine and sine transform are available
# whenever FFTW.jl is loaded together with LinearOperatorCollection:
using LinearOperatorCollection, FFTW
fop = FFTOp(Complex{eltype(image)}, shape = (N, N))
cop = DCTOp(eltype(image), shape = (N, N))
sop = DSTOp(eltype(image), shape = (N, N))
image_frequencies = reshape(fop * vec(image), N, N)
image_cosine = reshape(cop * vec(image), N, N)
image_sine = reshape(sop * vec(image), N, N)

fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], abs.(weighted_frequencies) .+ eps(), title = "Frequency Domain", colorscale = log10)
plot_image(fig[1,3], image_cosine, title = "Cosine")
plot_image(fig[1,4], image_sine, title = "Sine")
resize_to_layout!(fig)
fig