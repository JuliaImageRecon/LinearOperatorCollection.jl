# # GPU Acceleration
include("../../util.jl") #hide
# GPU kernels generally require all their arguments to exist on the GPU. This is not ncessarily the case for matrix-free operators as provides LinearOperators or LinearOperatorCollection.
# In the case that a matrix free operator is solely a function call and contains no internal array state, the operator is GPU compatible as long as the method has a GPU compatible implementation.

# If the operator has internal fields required for its computation, such as temporary arrays for intermediate values or indices, then it needs to move those to the GPU.
# Furthermore if the operator needs to create a new array in its execution, e.g. it is used in a non-inplace matrix-vector multiplication or it is combined with other operators, then the operator needs to specify
# a storage type. LinearOperatorCollection has several GPU compatible operators, where the storage type is given by setting a `S` parameter:
# ```julia
# using CUDA # or AMDGPU, Metal, ...
# image_gpu = cu(image)
# ```
using LinearOperatorCollection.LinearOperators
image_gpu = image #hide
storage = Complex.(similar(image_gpu, 0))
fop = FFTOp(eltype(image_gpu), shape = (N, N), S = typeof(storage))
LinearOperators.storage_type(fop) == typeof(storage)

# GPU operators can be used just like the other operators. Note however, that a GPU operator does not necessarily work with a CPU vector.