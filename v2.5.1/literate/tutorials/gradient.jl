# # Gradient Operator
include("../../util.jl") #hide
# This operator computes a direction gradient along one or more dimensions of an array:
gop = GradientOp(eltype(image); shape = (N, N), dims = 1)
gradients = reshape(gop * vec(image), :, N)
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], gradients[:, :], title = "Gradient", colormap = :vik)
resize_to_layout!(fig)
fig