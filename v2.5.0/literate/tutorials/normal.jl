# # Normal operator
include("../../util.jl") #hide

# This operator implements a lazy normal operator implementing:
# ```math
# \begin{equation}
#   (\mathbf{A})^*\mathbf{A}
# \end{equation}
# ```
# for some operator $\mathbf{A}$:
using FFTW
fop = op = FFTOp(ComplexF32, shape = (N, N))
nop = NormalOp(eltype(fop), parent = fop)
isapprox(nop * vec(image), vec(image))

# And we can again access our original operator:
typeof(nop.parent)

# LinearOperatorCollection also provides an opinionated `normalOperator` function which tries to optimize the resulting normal operator.
# As an example consider the normal operator of a weighted fourier operator:
weights = Float32.(collect(range(0, 1, length = N*N)))
wop = WeightingOp(weights)
pop = ProdOp(wop, fop)
nop = normalOperator(pop)
typeof(nop.parent) == typeof(pop)
# Note that the parent was changed. This is because the normal operator was optimized by initially computing the weights:
# ```math
# \begin{equation}
#   \tilde{\mathbf{W}} = \mathbf{W}^*\mathbf{W}
# \end{equation}
# ```
# and then applying the following each iteration:
# ```math
# \begin{equation}
#   \mathbf{A}^*\tilde{\mathbf{W}}\mathbf{A}
# \end{equation}
# ```

# Other operators can define different optimization strategies for the `normalOperator` method.