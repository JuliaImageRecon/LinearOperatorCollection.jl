# # Getting Started
# To begin, we first need to load LinearOperatorCollection:
using LinearOperatorCollection

# If we require an operator which is implemented via a package extensions, we also need to include
# the package that implements the functionality of the operator:
using FFTW

# As an introduction, we will construct a two dimensional FFT operator and apply it to an image.
# To construct an operator we can either call its constructor directly:
N = 256
op = FFTOp(ComplexF64, shape = (N, N))
typeof(op)

# Or we can use the factory method:
op = createLinearOperator(FFTOp{ComplexF64}, shape = (N, N))

# We will use a Shepp-logan phantom as an example image:
using ImagePhantoms, ImageGeoms
image = shepp_logan(N, SheppLoganToft())
size(image)

# Since our operators are only defined for matrix-vector products, we can't directly apply them to the two-dimensional image.
# We first have to reshape the image to a vector:
y = op * vec(image);

# Afterwards we can reshape the result and visualize it with CairoMakie:
image_freq = reshape(y, N, N)
using CairoMakie
function plot_image(figPos, img; title = "", width = 150, height = 150, colorscale = identity)
  ax = CairoMakie.Axis(figPos[1, 1]; yreversed=true, title, width, height)
  hidedecorations!(ax)
  hm = heatmap!(ax, img, colorscale = colorscale)
  Colorbar(figPos[1, 2], hm)
end
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], abs.(image_freq), title = "Frequency Domain", colorscale = log10)
resize_to_layout!(fig)
fig

# To perform the inverse Fourier transform we can simply use the adjoint of our operator:
image_inverse = reshape(adjoint(op) * y, N, N)
plot_image(fig[1,3], real.(image_inverse), title = "Image after Inverse")
resize_to_layout!(fig)
fig
