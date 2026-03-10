# # Radon Operator
include("../../util.jl") #hide
# The Radon operator is available when loading RadonKA.jl and LinearOperatorCollection:
using RadonKA
angles = collect(range(0, π, N))
rop = RadonOp(eltype(image); angles, shape = size(image));
sinogram = reshape(rop * vec(image), :, N)
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], sinogram, title = "Sinogram")
plot_image(fig[1,3], reshape(adjoint(rop) * vec(sinogram), N, N), title = "Backprojection")
resize_to_layout!(fig)
fig
