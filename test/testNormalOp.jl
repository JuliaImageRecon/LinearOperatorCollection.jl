
@info "test normal operator"
N = 512

Random.seed!(1234)
x = rand(N)
A = rand(N,N)
W = constructLinearOperator(WeightingOp{Float64}, weights=rand(N))

y1 = adjoint(A)*W*A*x
y = normalOperator(A,W)*x

@test norm(y1 - y) / norm(y) ≈ 0 atol=0.01

y1 = adjoint(A)*A*x
y = normalOperator(A)*x

@test norm(y1 - y) / norm(y) ≈ 0 atol=0.01