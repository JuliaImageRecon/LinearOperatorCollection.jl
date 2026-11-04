export RowBlockOp, rowblocks

mutable struct RowBlockOp{T, vecT, vecO, R, B, S} <: AbstractLinearOperator{T}
  const nrow :: Int
  const ncol :: Int
  const symmetric :: Bool
  const hermitian :: Bool
  const prod! :: Function
  const tprod! :: Function
  const ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  Mv :: vecT
  Mtu :: vecT
  const ops :: vecO
  const rows :: R
  const buffers :: B
  const scheduler :: S
end

LinearOperators.storage_type(::RowBlockOp{T, vecT}) where {T, vecT} = vecT

"""
    RowBlockOp(ops, rows; scheduler = DynamicScheduler())
    RowBlockOp(ops; scheduler = DynamicScheduler())
    RowBlockOp(A::AbstractMatrix, nblocks; mode = :interleaved, rng = Random.default_rng(), scheduler = DynamicScheduler())

create an operator out of the operators (or `Array`s) in `ops` stacked vertically, `[ops[1]; ops[2]; …]`, whose
rows are scattered over the rows of the result: the rows of `ops[k]` are the rows `rows[k]` of the operator. All
operators have the same number of columns, and `rows` is a partition of `1:nrow` into index vectors or ranges, one
per operator, of the length of that operator's number of rows. Without `rows`, the blocks are contiguous, in the
order of `ops`.

Block-iterative solvers (e.g. ordered-subset methods) access the blocks with [`rowblocks`](@ref). Applied as a
whole, the operator multiplies every block and writes its result into its rows, and its adjoint sums the adjoints
of the blocks applied to their rows. The products of the blocks run in parallel with `scheduler` (OhMyThreads.jl);
the adjoint sums the blocks one after the other.

Blocks with contiguous rows (`UnitRange`s) work on views of the data; other rows are gathered into and scattered
from a buffer per block (with index vectors on the device of the blocks, see `Adapt.adapt`). Reordering the data
such that every block is contiguous avoids these copies.

The third form splits the rows of a matrix into `nblocks` blocks, as copies of the rows: `:interleaved` takes every
`nblocks`-th row (`k:nblocks:m`), `:contiguous` consecutive rows, `:random` a random partition drawn with `rng`.
"""
function RowBlockOp(ops, rows; scheduler = DynamicScheduler())
  length(ops) == length(rows) || throw(ArgumentError("$(length(ops)) operators but $(length(rows)) row sets"))
  isempty(ops) && throw(ArgumentError("RowBlockOp needs at least one block"))
  ncol = size(first(ops), 2)
  all(op -> size(op, 2) == ncol, ops) || throw(DimensionMismatch("all blocks need the same number of columns"))
  for (op, r) in zip(ops, rows)
    length(r) == size(op, 1) || throw(DimensionMismatch("a block of $(size(op, 1)) rows has $(length(r)) row indices"))
  end
  nrow = sum(op -> size(op, 1), ops)
  sort!(reduce(vcat, [collect(r) for r in rows])) == 1:nrow ||
    throw(ArgumentError("the rows of the blocks must be a partition of 1:$nrow"))

  S = LinearOperators.storage_type(first(ops))
  for op in ops
    S = promote_type(S, LinearOperators.storage_type(op))
  end
  isconcretetype(S) || throw(LinearOperatorException("Storage types cannot be promoted to a concrete type"))

  # a buffer for every block whose rows are not contiguous
  buffers = [r isa AbstractUnitRange ? nothing : S(undef, length(r)) for r in rows]

  return RowBlockOp{eltype(first(ops)), S, typeof(ops), typeof(rows), typeof(buffers), typeof(scheduler)}(
    nrow, ncol, false, false,
    (res, x) -> rowBlockProd!(res, x, ops, rows, buffers, scheduler),
    (res, y) -> rowBlockAdjProd!(res, y, transpose, ops, rows, buffers),
    (res, y) -> rowBlockAdjProd!(res, y, adjoint, ops, rows, buffers),
    0, 0, 0, S(undef, 0), S(undef, 0),
    ops, rows, buffers, scheduler)
end

function RowBlockOp(ops; kwargs...)
  stops = cumsum([size(op, 1) for op in ops])
  starts = stops .- [size(op, 1) for op in ops] .+ 1
  return RowBlockOp(ops, [start:stop for (start, stop) in zip(starts, stops)]; kwargs...)
end

function RowBlockOp(A::AbstractMatrix, nblocks::Integer; mode::Symbol = :interleaved, rng = Random.default_rng(), kwargs...)
  m = size(A, 1)
  nblocks = clamp(nblocks, 1, m)
  rows = if mode == :interleaved
    [k:nblocks:m for k in 1:nblocks]
  elseif mode == :contiguous
    [(div((k - 1) * m, nblocks) + 1):div(k * m, nblocks) for k in 1:nblocks]
  elseif mode == :random
    perm = randperm(rng, m)
    [sort(perm[k:nblocks:m]) for k in 1:nblocks]
  else
    throw(ArgumentError("mode must be :interleaved, :contiguous or :random, got :$mode"))
  end
  return RowBlockOp([A[r, :] for r in rows], rows; kwargs...)
end

"""
    rowblocks(op::RowBlockOp)

the blocks of `op` as pairs `(operator, rows)`.
"""
rowblocks(op::RowBlockOp) = collect(zip(op.ops, op.rows))

# the rows r of a data vector y: a view for contiguous rows, else gathered into the buffer
@inline block_data(y, r::AbstractUnitRange, ::Nothing) = view(y, r)
@inline block_data(y, r, buffer) = (buffer .= view(y, r); buffer)

function rowBlockProd!(y, x, ops, rows, buffers, scheduler)
  @tasks for k in 1:length(ops)
    @set scheduler = scheduler
    r, buffer = rows[k], buffers[k]
    if isnothing(buffer)
      mul!(view(y, r), ops[k], x)
    else
      mul!(buffer, ops[k], x)
      view(y, r) .= buffer
    end
  end
  return y
end

function rowBlockAdjProd!(x, y, adj, ops, rows, buffers)
  for k in 1:length(ops)
    yk = block_data(y, rows[k], buffers[k])
    if k == 1
      mul!(x, adj(ops[k]), yk)
    else
      mul!(x, adj(ops[k]), yk, true, true)
    end
  end
  return x
end

# Index vectors move to the device of the adapted blocks and stay integers (adapting them with `to`
# itself could convert them, e.g. with `to = CuArray{Float32}`).
function Adapt.adapt_structure(to, op::RowBlockOp)
  ops = map(child -> Adapt.adapt(to, child), op.ops)
  S = LinearOperators.storage_type(first(ops))
  rows = map(r -> r isa AbstractUnitRange ? r : copyto!(similar(S(undef, 0), eltype(r), length(r)), collect(r)), op.rows)
  return RowBlockOp(ops, rows; scheduler = op.scheduler)
end

### Normal Matrix Code ###

mutable struct RowBlockNormalOp{T, vecT, V} <: AbstractLinearOperator{T}
  const nrow :: Int
  const ncol :: Int
  const symmetric :: Bool
  const hermitian :: Bool
  const prod! :: Function
  const tprod! :: Nothing
  const ctprod! :: Nothing
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  Mv :: vecT
  Mtu :: vecT
  normalOps :: V
  tmp :: vecT
end

LinearOperators.storage_type(::RowBlockNormalOp{T, vecT}) where {T, vecT} = vecT

function RowBlockNormalOp(normalOps, tmp::vecT) where {vecT}
  n = size(first(normalOps), 1)
  return RowBlockNormalOp{eltype(first(normalOps)), vecT, typeof(normalOps)}(n, n, false, false,
    (res, x) -> rowBlockNormalProd!(res, normalOps, tmp, x), nothing, nothing,
    0, 0, 0, vecT(undef, 0), vecT(undef, 0), normalOps, tmp)
end

function rowBlockNormalProd!(y, normalOps, tmp, x)
  mul!(y, normalOps[1], x)
  for k in 2:length(normalOps)
    mul!(tmp, normalOps[k], x)
    y .+= tmp
  end
  return y
end

function Adapt.adapt_structure(to, op::RowBlockNormalOp)
  return RowBlockNormalOp(map(child -> Adapt.adapt(to, child), op.normalOps), Adapt.adapt(to, op.tmp))
end

"""
    normalOperator(op::RowBlockOp, W = nothing; kwargs...)

the normal operator `adjoint(op) * W * op` as the sum of the normal operators of the blocks (each with the part of
the weights `W` that belongs to its rows), so that specialized normal operators of the blocks are used.
"""
function LinearOperatorCollection.normalOperator(op::RowBlockOp, W = nothing; kwargs...)
  S = LinearOperators.storage_type(op)
  if isnothing(W)
    normalOps = [normalOperator(o; kwargs...) for o in op.ops]
  else
    T = promote_type(eltype(op), eltype(W))
    weights = W * fill!(S(undef, op.nrow), one(eltype(op)))
    normalOps = [normalOperator(o, WeightingOp(T; weights = T.(weights[r])); kwargs...) for (o, r) in zip(op.ops, op.rows)]
  end
  return RowBlockNormalOp(normalOps, S(undef, op.ncol))
end
