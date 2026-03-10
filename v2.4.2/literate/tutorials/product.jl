# # Product Operator
include("../../util.jl") #hide
# This operator describes the product or composition between two operators:
weights = collect(range(0, 1, length = N*N))
wop = WeightingOp(weights)
fop = FFTOp(ComplexF64, shape = (N, N));
# A feature of LinearOperators.jl is that operator can be cheaply transposed, conjugated and multiplied and only in the case of a matrix-vector product the combined operation is evaluated.
tmp_op = wop * fop
tmp_freqs = tmp_op * vec(image)


# Similar to the WeightingOp, the main difference with the product operator provided by LinearOperatorCollection is the dedicated type, which allows for code specialisation.
pop = ProdOp(wop, fop)
typeof(pop)
# and the ability to retrieve the components:
typeof(pop.A)
# and
typeof(pop.B)

# Otherwise they compute the same thing:
pop * vec(image) == tmp_op * vec(image)

# Note that a product operator is not thread-safe.

# We can again visualize our result:
weighted_frequencies = reshape(pop * vec(image), N, N)
image_inverse = reshape(adjoint(pop) * vec(weighted_frequencies), N, N)
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], abs.(weighted_frequencies) .+ eps(), title = "Frequency Domain", colorscale = log10)
plot_image(fig[1,3], real.(image_inverse), title = "Image after Inverse")
resize_to_layout!(fig)
fig