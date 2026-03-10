# # Block Diagonal Operator
include("../../util.jl") #hide
# This operator represents a block-diagonal matrix out of given operators.
# One can also provide a single-operator and a number of blocks. In that case the given operator is repeated for each block.
# In the case of stateful operators, one can supply a method for copying the operators.
blocks = N
ops = [WeightingOp(fill(i % 2, N)) for i = 1:N]
dop = DiagOp(ops)
typeof(dop)

# We can retrieve the operators:
typeof(dop.ops)

# And visualize the result:
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], reshape(dop * vec(image), N, N), title = "Block Weighted")
resize_to_layout!(fig)
fig

# The default operator is created with a DynamicScheduler from OhMyThreads.jl. This means it will execute the multiplication of its
# individual blocks in parallel. To supply a different scheduler do:
using OhMyThreads
scheduler = SerialScheduler()
dop_serial = DiagOp(ops, scheduler = scheduler)
typeof(dop_serial)