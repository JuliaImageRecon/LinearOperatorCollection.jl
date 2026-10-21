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

function initialize_normal_buffers!(op)
  op.y .= rand(eltype(op.y), length(op.y))
  for normal_op in op.normalOps
    normal_op.tmp .= rand(eltype(normal_op.tmp), length(normal_op.tmp))
  end
end

function check_normal_buffers(op, snapshot)
  @test op.y == snapshot.y
  @test all(normal_op.tmp == expected for (normal_op, expected) in zip(op.normalOps, snapshot.tmps))
end

function snapshot_normal_children(op)
  parents = map(normal_op -> begin
    parent = normal_op.parent
    parent isa WeightingOp ? copy(parent.weights) : copy(parent)
  end, op.normalOps)
  weights = map(normal_op -> copy(normal_op.weights.weights), op.normalOps)
  return (parents = parents, weights = weights)
end

function check_normal_children(op, snapshot)
  for (normal_op, expected) in zip(op.normalOps, snapshot.parents)
    parent = normal_op.parent
    if parent isa WeightingOp
      @test parent.weights == expected
    else
      @test parent == expected
    end
  end
  @test all(normal_op.weights.weights == expected for (normal_op, expected) in zip(op.normalOps, snapshot.weights))
end

@testset "Adapt reconstruction" begin
  N = 16

  # CPU tests use JLArray; GPU jobs add their configured target without
  # repeating an Array target already supplied by runtests.jl.
  for array_type in unique([JLArray; filter(T -> T != Array, arrayTypes)])
    @testset "target storage $(array_type)" begin
      @testset "WeightingOp" begin
        weights = rand(Float32, N)
        source = WeightingOp(weights, 2)
        source_weights = copy(source.weights)
        adapted = Adapt.adapt(array_type, source)

        @test source.weights == source_weights
        check_adapted_storage(adapted, array_type)
        @test typeof(adapted.weights) <: array_type
        @test typeof(adapted.op.Mv) <: array_type
        @test typeof(source.weights) <: Array
        check_adapted_mul(source, adapted, rand(Float32, 2N), array_type)
      end

      @testset "ProdOp, NormalOpImpl, and ProdNormalOp recurse" begin
        left = WeightingOp(rand(Float32, N))
        right = WeightingOp(rand(Float32, N))
        parent = ProdOp(left, right)
        weights = WeightingOp(rand(Float32, N))
        source_prod = ProdOp(left, parent)
        source_normal = LinearOperatorCollection.NormalOpImpl(source_prod, weights)
        source_prod_normal = normalOperator(source_prod)

        # Initialize every non-empty source scratch buffer before taking the
        # snapshots. Adapt must not mutate any of these host-side values.
        for tmp in (parent.tmp, source_prod.tmp, source_normal.tmp,
                    source_prod_normal.tmp, source_prod_normal.normalOpInner.tmp)
          tmp .= rand(Float32, length(tmp))
        end
        source_snapshot = (
          left_weights = copy(left.weights),
          right_weights = copy(right.weights),
          weights = copy(weights.weights),
          parent_tmp = copy(parent.tmp),
          source_prod_tmp = copy(source_prod.tmp),
          source_normal_tmp = copy(source_normal.tmp),
          source_prod_normal_tmp = copy(source_prod_normal.tmp),
          source_prod_normal_inner_tmp = copy(source_prod_normal.normalOpInner.tmp),
          source_prod_normal_inner_weights = copy(source_prod_normal.normalOpInner.weights.weights),
        )

        adapted_prod = Adapt.adapt(array_type, source_prod)
        adapted_normal = Adapt.adapt(array_type, source_normal)
        adapted_prod_normal = Adapt.adapt(array_type, source_prod_normal)

        # Check immutability before any multiplication can legitimately
        # update the source scratch buffers.
        @test left.weights == source_snapshot.left_weights
        @test right.weights == source_snapshot.right_weights
        @test weights.weights == source_snapshot.weights
        @test parent.tmp == source_snapshot.parent_tmp
        @test source_prod.tmp == source_snapshot.source_prod_tmp
        @test source_normal.tmp == source_snapshot.source_normal_tmp
        @test source_prod_normal.tmp == source_snapshot.source_prod_normal_tmp
        @test source_prod_normal.normalOpInner.tmp == source_snapshot.source_prod_normal_inner_tmp
        @test source_prod_normal.normalOpInner.weights.weights == source_snapshot.source_prod_normal_inner_weights

        check_adapted_storage(adapted_prod, array_type)
        @test typeof(adapted_prod.tmp) <: array_type
        @test typeof(adapted_prod.B.tmp) <: array_type
        @test typeof(adapted_prod.B.A.weights) <: array_type

        check_adapted_storage(adapted_normal, array_type)
        @test typeof(adapted_normal.tmp) <: array_type
        check_adapted_storage(adapted_normal.parent, array_type)
        check_adapted_storage(adapted_normal.weights, array_type)

        check_adapted_storage(adapted_prod_normal, array_type)
        @test typeof(adapted_prod_normal.tmp) <: array_type
        check_adapted_storage(adapted_prod_normal.opOuter, array_type)
        check_adapted_storage(adapted_prod_normal.normalOpInner, array_type)
        if adapted_prod_normal.opOuter isa WeightingOp
          @test typeof(adapted_prod_normal.opOuter.weights) <: array_type
        else
          @test typeof(adapted_prod_normal.opOuter.tmp) <: array_type
        end
        @test typeof(adapted_prod_normal.normalOpInner.tmp) <: array_type

        x = rand(Float32, N)
        check_adapted_mul(source_prod, adapted_prod, x, array_type)
        check_adapted_mul(source_normal, adapted_normal, x, array_type)
        check_adapted_mul(source_prod_normal, adapted_prod_normal, x, array_type)
      end

      @testset "DiagOp and DiagNormalOp recurse" begin
        blocks = [WeightingOp(rand(Float32, N)), WeightingOp(rand(Float32, N))]
        sources = (DiagOp(blocks), DiagOp(rand(Float32, N, N), 2))

        for source in sources
          source_ops_snapshot = map(block -> block isa WeightingOp ? copy(block.weights) : copy(block), source.ops)
          source_normal = normalOperator(source)
          initialize_normal_buffers!(source_normal)
          source_normal_snapshot = (
            y = copy(source_normal.y),
            tmps = map(normal_op -> copy(normal_op.tmp), source_normal.normalOps),
          )
          source_normal_children_snapshot = snapshot_normal_children(source_normal)

          adapted = Adapt.adapt(array_type, source)
          adapted_normal = Adapt.adapt(array_type, source_normal)

          # All source data was initialized before the snapshots and is
          # checked before either source operator is multiplied.
          @test all(begin
            block isa WeightingOp ? block.weights == expected : block == expected
          end for (block, expected) in zip(source.ops, source_ops_snapshot))
          check_normal_buffers(source_normal, source_normal_snapshot)
          check_normal_children(source_normal, source_normal_children_snapshot)

          check_adapted_storage(adapted, array_type)
          @test adapted.equalOps == source.equalOps
          @test typeof(adapted.Mv) <: array_type
          @test all(begin
            S = LinearOperators.storage_type(block)
            isconcretetype(S) && S <: array_type
          end for block in adapted.ops)
          @test typeof(source.Mv) <: Array

          check_adapted_storage(adapted_normal, array_type)
          @test adapted_normal.nrow == source_normal.nrow
          @test adapted_normal.ncol == source_normal.ncol
          @test typeof(adapted_normal.idx) == typeof(source_normal.idx)
          @test adapted_normal.idx == source_normal.idx
          @test adapted_normal.scheduler === source_normal.scheduler
          @test typeof(adapted_normal.y) <: array_type
          @test all(begin
            check_adapted_storage(normal_op, array_type)
            typeof(normal_op.tmp) <: array_type
          end for normal_op in adapted_normal.normalOps)

          x = rand(Float32, 2N)
          constructed_normal = normalOperator(adapted)
          check_adapted_storage(constructed_normal, array_type)
          check_adapted_mul(source, adapted, x, array_type)
          check_adapted_mul(source_normal, adapted_normal, x, array_type)
          check_adapted_mul(source_normal, constructed_normal, x, array_type)
        end
      end

      @testset "DiagOp preserves tuple ops" begin
        source = DiagOp((WeightingOp(rand(Float32, N)), WeightingOp(rand(Float32, N))))
        source_ops_snapshot = map(block -> copy(block.weights), source.ops)
        source_normal = normalOperator(source)
        initialize_normal_buffers!(source_normal)
        source_normal_snapshot = (
          y = copy(source_normal.y),
          tmps = map(normal_op -> copy(normal_op.tmp), source_normal.normalOps),
        )
        source_normal_children_snapshot = snapshot_normal_children(source_normal)

        adapted = Adapt.adapt(array_type, source)
        adapted_normal = Adapt.adapt(array_type, source_normal)

        @test all(block.weights == expected for (block, expected) in zip(source.ops, source_ops_snapshot))
        check_normal_buffers(source_normal, source_normal_snapshot)
        check_normal_children(source_normal, source_normal_children_snapshot)

        @test adapted.ops isa Tuple
        @test typeof(adapted.ops) <: Tuple
        @test length(adapted.ops) == length(source.ops)
        @test all(typeof(block.weights) <: array_type for block in adapted.ops)
        check_adapted_storage(adapted, array_type)
        check_adapted_storage(adapted_normal, array_type)
        @test typeof(adapted_normal.y) <: array_type
        @test all(typeof(normal_op.tmp) <: array_type for normal_op in adapted_normal.normalOps)

        x = rand(Float32, 2N)
        check_adapted_mul(source, adapted, x, array_type)
        check_adapted_mul(source_normal, adapted_normal, x, array_type)
      end
    end
  end
end
