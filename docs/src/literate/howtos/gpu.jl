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

# ## Moving an Operator to Another Backend
# Operators that implement [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) support can be moved to another array backend with `Adapt.adapt(target, op)`.
# Which backends are supported depends on the operator and the target, and not every operator in LinearOperatorCollection implements Adapt.

# The target is the backing array type without element-type parameters, such as `Array`, `CuArray` or `ROCArray`.
# LinearOperatorCollection does not support element-typed targets such as `Array{ComplexF64}` or `CuArray{Float32}`, even when the element type matches the operator.
# Adapting moves an operator to another backend; it is not the way to change the operator's precision. Choose the element type when you construct the operator.

# This differs from the `S` keyword above, which takes a parameterized storage type such as `CuArray{T, 1}`:
# ```julia
# using LinearOperatorCollection, Adapt
# op = GradientOp(Float32; shape = (32, 32))
# op_cpu = Adapt.adapt(Array, op)    # not Array{Float32}
#
# using CUDA                         # requires CUDA.jl and a supported NVIDIA GPU
# op_gpu = Adapt.adapt(CuArray, op)  # not CuArray{Float32}
# ```
# GPU targets require the corresponding backend package, such as CUDA.jl or AMDGPU.jl, and supported hardware.
