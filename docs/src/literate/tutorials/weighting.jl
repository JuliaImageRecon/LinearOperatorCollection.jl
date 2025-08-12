# # Weighting Operator
include("../../util.jl") #hide
# The weighting operator implements a diagonal matrix which multiplies a vector index-wise with given weights.
# Such an operator is also implemented within LinearOperator.jl, however here this operator has a dedicated type on which one can dispatch:
weights = collect(range(0, 1, length = N*N))
op = WeightingOp(weights)

# Such an operator is also implemented within LinearOperator.jl, however here this operator has a dedicated type on which one can dispatch:
typeof(op)

# And it is possible to retrieve the weights from the operator
op.weights == weights

# Note that we didn't need to specify the element type of the operator here. In this case the eltype was derived from the provided weights.
weighted_image = reshape(op * vec(image), N, N)

# To visualize our weighted image, we will again use CairoMakie:
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], weighted_image, title = "Weighted Image")
resize_to_layout!(fig)
fig