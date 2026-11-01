# # Row-Block Operator
include("../../util.jl") #hide
# This operator stacks operators vertically: all of them act on the same input, and each one fills its own rows of
# the output. The rows of a block need not be contiguous. Block-iterative and ordered-subset methods (e.g. SART,
# OS-EM) use such a split of a system matrix into blocks of rows; as a whole the operator works like any other.
#
# A matrix split into three blocks of interleaved rows (every third row), `:contiguous` and `:random` are the other
# modes:
A = rand(12, N * N)
rop = RowBlockOp(A, 3)
typeof(rop)

# The blocks with the indices of their rows:
[(size(op), rows) for (op, rows) in rowblocks(rop)]

# Applied as a whole, the operator equals the matrix, and so do its adjoint and its normal operator:
x = vec(image)
rop * x ≈ A * x

#-
adjoint(rop) * (A * x) ≈ adjoint(A) * (A * x)

# Any operators with the same number of columns can be blocks, with their rows given explicitly. Here two weighting
# operators fill the even and the odd entries of the output:
ops = [WeightingOp(fill(1.0, N * N)), WeightingOp(fill(2.0, N * N))]
rop2 = RowBlockOp(ops, [1:2:2N*N, 2:2:2N*N])
(rop2 * x)[1:4] ≈ [x[1], 2x[1], x[2], 2x[2]]

# Blocks with contiguous rows work on views of the data; for other rows the data of a block is gathered into a
# buffer. The products of the blocks run in parallel with a scheduler of OhMyThreads.jl, as for the
# [`DiagOp`](@ref):
using OhMyThreads
rop_serial = RowBlockOp(A, 3; scheduler = SerialScheduler())
typeof(rop_serial)
