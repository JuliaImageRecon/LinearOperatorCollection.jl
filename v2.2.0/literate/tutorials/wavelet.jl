# # Wavelet Operator
include("../../util.jl") #hide
# The wavelet operator is available when loading Wavelets.jl together with LinearOperatorCollection:
using Wavelets
wop = WaveletOp(eltype(image), shape = (N, N))
wavelet_image = reshape(wop * vec(image), N, N)
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], abs.(wavelet_image) .+ eps(), title = "Wavelet Image", colorscale = sqrt)
resize_to_layout!(fig)
fig
