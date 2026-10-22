function check_adapted_storage(op, array_type)
  S = LinearOperators.storage_type(op)
  @test isconcretetype(S)
  @test S <: array_type
end

function check_adapted_mul(source, adapted, x, array_type)
  y_source = zeros(eltype(x), size(source, 1))
  y_adapted = array_type(zeros(eltype(x), size(adapted, 1)))
  mul!(y_source, source, x)
  mul!(y_adapted, adapted, array_type(x))
  @test Array(y_adapted) ≈ y_source rtol = 1f-4
end

@testset "Adapt reconstruction" begin
  N = 16
  array_type = JLArray

  @testset "WeightingOp" begin
    weights = rand(Float32, N)
    source = WeightingOp(weights, 2)
    adapted = Adapt.adapt(array_type, source)

    check_adapted_storage(adapted, array_type)
    @test typeof(adapted.weights) <: array_type
    @test typeof(adapted.op.Mv) <: array_type
    @test typeof(source.weights) <: Array
    check_adapted_mul(source, adapted, rand(Float32, 2N), array_type)
  end

  @testset "ProdOp and NormalOpImpl recurse" begin
    left = WeightingOp(rand(Float32, N))
    right = WeightingOp(rand(Float32, N))
    parent = ProdOp(left, right)
    weights = WeightingOp(rand(Float32, N))
    source_prod = ProdOp(left, parent)
    source_normal = LinearOperatorCollection.NormalOpImpl(source_prod, weights)

    adapted_prod = Adapt.adapt(array_type, source_prod)
    adapted_normal = Adapt.adapt(array_type, source_normal)

    check_adapted_storage(adapted_prod, array_type)
    @test typeof(adapted_prod.tmp) <: array_type
    @test typeof(adapted_prod.B.tmp) <: array_type
    @test typeof(adapted_prod.B.A.weights) <: array_type
    check_adapted_storage(adapted_normal, array_type)
    @test typeof(adapted_normal.tmp) <: array_type
    check_adapted_storage(adapted_normal.parent, array_type)
    check_adapted_storage(adapted_normal.weights, array_type)
    @test typeof(source_prod.tmp) <: Array
    @test typeof(source_normal.tmp) <: Array

    x = rand(Float32, N)
    check_adapted_mul(source_prod, adapted_prod, x, array_type)
    check_adapted_mul(source_normal, adapted_normal, x, array_type)
  end

  @testset "DiagOp preserves equalOps" begin
    blocks = [WeightingOp(rand(Float32, N)), WeightingOp(rand(Float32, N))]
    sources = (DiagOp(blocks), DiagOp(rand(Float32, N, N), 2))

    for source in sources
      adapted = Adapt.adapt(array_type, source)
      check_adapted_storage(adapted, array_type)
      @test adapted.equalOps == source.equalOps
      @test typeof(adapted.Mv) <: array_type
      @test all(begin
        S = LinearOperators.storage_type(block)
        isconcretetype(S) && S <: array_type
      end for block in adapted.ops)
      @test typeof(source.Mv) <: Array

      x = rand(Float32, 2N)
      check_adapted_mul(source, adapted, x, array_type)

      source_normal = normalOperator(source)
      adapted_normal = normalOperator(adapted)
      check_adapted_storage(adapted_normal, array_type)
      check_adapted_mul(source_normal, adapted_normal, x, array_type)
    end
  end
end
