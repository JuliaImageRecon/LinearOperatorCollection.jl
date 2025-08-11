# # Block Diagonal Operator
# This operator represents a block-diagonal matrix out of given operators.
# One can also provide a single-operator and a number of blocks. In that case the given operator is repeated for each block.
# In the case of stateful operators, one can supply a method for copying the operators.
blocks = N
ops = [WeightingOp(fill(i % 2, N)) for i = 1:N]
dop = DiagOp(ops)
typeof(dop)

# We can retrieve the operators:
dop.ops

# And visualize the result:
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], reshape(dop * vec(image), N, N), title = "Block Weighted")
resize_to_layout!(fig)
fig