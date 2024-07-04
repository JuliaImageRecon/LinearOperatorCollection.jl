
@testset "Normal Operator" begin
  for arrayType in arrayTypes
    @testset "$arrayType" begin 
      N = 512

      Random.seed!(1234)
      x = arrayType(rand(N))
      A = arrayType(rand(N,N))
      A_adj = arrayType(collect(adjoint(A))) # LinearOperators can't resolve storage_type otherwise
      W = WeightingOp(arrayType(rand(N)))
      WA = W*A

      y1 = Array(A_adj*W*W*A*x)
      y2 = Array(adjoint(WA) * WA * x)
      y = Array(normalOperator(A,W)*x)

      @test norm(y1 - y) / norm(y) ≈ 0 atol=0.01
      @test norm(y2 - y) / norm(y) ≈ 0 atol=0.01


      y1 = Array(adjoint(A)*A*x)
      y = Array(normalOperator(A)*x)

      @test norm(y1 - y) / norm(y) ≈ 0 atol=0.01
    end
  end
end