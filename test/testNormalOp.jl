
@testset "Normal Operator" begin
  for arrayType in arrayTypes
    for elType in [Float32, ComplexF32]
      @testset "$arrayType" begin 
        N = 512

        Random.seed!(1234)
        x = arrayType(rand(elType, N))
        A = arrayType(rand(elType, N,N))
        A_adj = arrayType(collect(adjoint(A))) # LinearOperators can't resolve storage_type otherwise
        W = WeightingOp(arrayType(rand(elType, N)))
        WA = W*A
        WHW = adjoint.(W.weights) .* W.weights
        prod = ProdOp(W, A)

        y1 = Array(A_adj*adjoint(W)*W*A*x)
        y2 = Array(adjoint(WA) * WA * x)
        y3 = Array(normalOperator(prod) * x)
        y4 = Array(normalOperator(A, WHW)*x)

        @test norm(y1 - y4) / norm(y4) ≈ 0 atol=0.01
        @test norm(y2 - y4) / norm(y4) ≈ 0 atol=0.01
        @test norm(y3 - y4) / norm(y4) ≈ 0 atol=0.01


        y1 = Array(adjoint(A)*A*x)
        y = Array(normalOperator(A)*x)

        @test norm(y1 - y) / norm(y) ≈ 0 atol=0.01

        B = arrayType(rand(elType, N, N))
        AB = A * B
        WAB = diagm(W.weights)*A*B
        @test isapprox(normalOperator(ProdOp(A, B)) * x, adjoint(AB) * AB * x)
        @test isapprox(normalOperator(ProdOp(ProdOp(W, A), B)) * x, adjoint(WAB) * WAB * x)
      end
    end
  end
end