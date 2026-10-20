function check_adapted_storage(op)
  S = LinearOperators.storage_type(op)
  @test isconcretetype(S)
  @test S <: JLArray
end

function check_adapted_mul(source, adapted, x)
  y_source = zeros(eltype(x), size(source, 1))
  y_adapted = JLArray(zeros(eltype(x), size(adapted, 1)))
  mul!(y_source, source, x)
  mul!(y_adapted, adapted, JLArray(x))
  @test Array(y_adapted) ≈ y_source rtol = 1f-4
end

@testset "Adapt reconstruction" begin
  N = 16

  @testset "WeightingOp" begin
    weights = rand(Float32, N)
    source = WeightingOp(weights, 2)
    adapted = Adapt.adapt(JLArray, source)

    check_adapted_storage(adapted)
    @test typeof(adapted.weights) <: JLArray
    @test typeof(adapted.op.Mv) <: JLArray
    @test typeof(source.weights) <: Array
    check_adapted_mul(source, adapted, rand(Float32, 2N))
  end

  @testset "ProdOp and NormalOpImpl recurse" begin
    left = WeightingOp(rand(Float32, N))
    right = WeightingOp(rand(Float32, N))
    parent = ProdOp(left, right)
    weights = WeightingOp(rand(Float32, N))
    source_prod = ProdOp(left, parent)
    source_normal = LinearOperatorCollection.NormalOpImpl(source_prod, weights)

    adapted_prod = Adapt.adapt(JLArray, source_prod)
    adapted_normal = Adapt.adapt(JLArray, source_normal)

    check_adapted_storage(adapted_prod)
    @test typeof(adapted_prod.tmp) <: JLArray
    @test typeof(adapted_prod.B.tmp) <: JLArray
    @test typeof(adapted_prod.B.A.weights) <: JLArray
    check_adapted_storage(adapted_normal)
    @test typeof(adapted_normal.tmp) <: JLArray
    check_adapted_storage(adapted_normal.parent)
    check_adapted_storage(adapted_normal.weights)
    @test typeof(source_prod.tmp) <: Array
    @test typeof(source_normal.tmp) <: Array

    x = rand(Float32, N)
    check_adapted_mul(source_prod, adapted_prod, x)
    check_adapted_mul(source_normal, adapted_normal, x)
  end

  @testset "DiagOp preserves equalOps" begin
    blocks = [WeightingOp(rand(Float32, N)), WeightingOp(rand(Float32, N))]
    sources = (DiagOp(blocks), DiagOp(rand(Float32, N, N), 2))

    for source in sources
      adapted = Adapt.adapt(JLArray, source)
      check_adapted_storage(adapted)
      @test adapted.equalOps == source.equalOps
      @test typeof(adapted.Mv) <: JLArray
      @test all(begin
        S = LinearOperators.storage_type(block)
        isconcretetype(S) && S <: JLArray
      end for block in adapted.ops)
      @test typeof(source.Mv) <: Array

      x = rand(Float32, 2N)
      check_adapted_mul(source, adapted, x)

      source_normal = normalOperator(source)
      adapted_normal = normalOperator(adapted)
      check_adapted_storage(adapted_normal)
      check_adapted_mul(source_normal, adapted_normal, x)
    end
  end
end
