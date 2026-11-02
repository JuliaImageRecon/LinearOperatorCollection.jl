"""
  GradientOp(T::Type; shape::Tuple, dims=1:length(shape))

directional gradient operator along the dimensions `dims` for an array of size `shape`.

# Required Argument
  * `T`                        - type of elements, .e.g. `Float64` for `ComplexF32`

# Required Keyword argument
  * `shape::NTuple{N,Int}`     - shape of the array (e.g., image)

# Optional Keyword argument
  * `dims`                     - dimension(s) along which the gradient is applied; default is `1:length(shape)`
  * `S`                        - storage type of the operator, e.g. `Vector{T}` (default) or `CuArray{T, 1}`

# Moving to another array backend

    Adapt.adapt(CuArray, op)

returns a new gradient operator for the target array type (e.g. `Array`, `JLArray`, `CuArray`
or `ROCArray`), and `copy(op)` returns a new one with the same storage type. Both rebuild the
operator and its temporary storage from `shape` and the directions taken from `dims` at
construction; a later change of the vector passed as `dims`, or an iterator consumed by the
construction, has no effect. `op` itself is not modified. The directions stay on the host.
The adapted storage must have the element type of `op`: an element-typed target such as
`CuArray{Float32}` for a `Float64` operator throws an `ArgumentError`, and so does a target
without an element type for an operator built with an `S` of another element type.
"""
function GradientOp(::Type{T}; shape::NTuple{N,Int}, dims=1:length(shape), kwargs...) where {T <: Number, N}
  return GradientOpImpl(T, shape, dims; kwargs...)
end

struct GradientOpImpl{T, vecT <: AbstractVector, O <: AbstractLinearOperator{T}, D, N, W} <: GradientOp{T}
  op::O
  shape::NTuple{N, Int}
  # `Int` for a single direction, otherwise the tuple of directions taken from `dims` at construction
  dims::D
  # temporary vector of the adjoint for several directions, `nothing` for a single direction
  tmp::W
  function GradientOpImpl(op::O, shape::NTuple{N, Int}, dims::D, tmp::W) where {T, O <: AbstractLinearOperator{T}, D, N, W}
    return new{T, storage_type(op), O, D, N, W}(op, shape, dims, tmp)
  end
end

function GradientOpImpl(T::Type, shape::NTuple{N,Int}, dims; S = Vector{T}) where N
  # Iterate `dims` once: replaying the operator must not depend on the caller's container later on
  dims = Tuple(dims)
  dirs = CartesianIndex{N}[]
  cols = Int64[]
  for dim in dims
    δ = zeros(Int32, N)
    δ[dim] = 1
    δ = NTuple{N}(δ)
    di = CartesianIndex(δ)
    push!(dirs, di)
    push!(cols, div((shape[dim]-1)*prod(shape), shape[dim]))
  end
  dim_ends = accumulate(+, cols)
  
  nrow = sum(cols)
  ncol = prod(shape)

  tmp = S(undef, ncol)
  
  op = LinearOperator{T}(nrow, ncol, false, false,
                          (res,x) -> (grad!(res,x,shape,dirs, dims, dim_ends)),
                          (res,x) -> (grad_t!(res,x,shape,dirs, dims, dim_ends, tmp)),
                          (res,x) -> (grad_t!(res,x,shape,dirs, dims, dim_ends, tmp)),
                          S = S)
  return GradientOpImpl(op, shape, dims, tmp)
end

function GradientOpImpl(T::Type, shape::NTuple{N,Int}, dim::Int; S = Vector{T}) where N
  nrow = div( (shape[dim]-1)*prod(shape), shape[dim] )
  ncol = prod(shape)
  δ = zeros(Int, length(shape))
  δ[dim] = 1
  δ = Tuple(δ)
  dir = CartesianIndex(δ)
  op = LinearOperator{T}(nrow, ncol, false, false,
                          (res,x) -> (grad!(res,x,shape,dir)),
                          (res,x) -> (grad_t!(res,x,shape,dir)),
                          (res,x) -> (grad_t!(res,x,shape,dir)),
                          S = S)
  return GradientOpImpl(op, shape, dim, nothing)
end

function Base.getproperty(op::GradientOpImpl, field::Symbol)
  if in(field, (:op, :shape, :dims, :tmp))
    return getfield(op, field)
  else
    return getproperty(getfield(op, :op), field)
  end
end
Base.setproperty!(op::GradientOpImpl, field::Symbol, value) = setproperty!(op.op, field, value)

storage_type(::GradientOpImpl{T, vecT}) where {T, vecT} = vecT

# Rebuild the operator, its closures and temporary vector from the shape and the directions
_rebuild_gradient(op::GradientOpImpl, S) = GradientOpImpl(eltype(op), op.shape, op.dims; S = S)

Base.copy(op::GradientOpImpl) = _rebuild_gradient(op, storage_type(op))

function Adapt.adapt_structure(to, op::GradientOpImpl)
  target_storage = _adapt_target_storage(to, op.Mv, "GradientOp")
  # The rebuild keeps the element type of the operator, so the target storage must share it
  eltype(target_storage) == eltype(op) || throw(ArgumentError(
    "unsupported Adapt target $(to) for GradientOp: storage element type $(eltype(target_storage)) " *
    "differs from the operator's element type $(eltype(op))"))
  return _rebuild_gradient(op, typeof(target_storage))
end

function grad!(res::T, img::U, shape, dirs, dims, dim_ends) where {T<:AbstractVector, U<:AbstractVector}
  dim_start = 1

  for (i, dir) in enumerate(dirs)
    grad!(view(res, dim_start:dim_ends[i]), img, shape, dir)
    dim_start = dim_ends[i] + 1
  end
end

# directional gradients
function grad!(res::T, img::U, shape::NTuple{N,Int64}, di::CartesianIndex{N}) where {N, T<:AbstractVector, U<:AbstractVector}
  img_ = reshape(img,shape)

  res_ = reshape(res, shape .- Tuple(di))

  @tasks for i ∈ CartesianIndices(res_)
    @inbounds res_[i] = img_[i] - img_[i + di]
  end
end

function grad_t!(res::T, g::U, shape, dirs, dims, dims_end, tmp) where {T<:AbstractVector, U<:AbstractVector}
  dim_start = 1

  fill!(res, zero(eltype(res)))
  for (i, dir) in enumerate(dirs)
    grad_t!(tmp, view(g, dim_start:dims_end[i]), shape, dir)
    dim_start = dims_end[i] + 1
    res .= res .+ tmp 
  end
end

# adjoint of directional gradients
function grad_t!(res::T, g::U, shape::NTuple{N,Int64}, di::CartesianIndex{N}) where {N, T<:AbstractVector, U<:AbstractVector}
  res_ = reshape(res,shape)
  g_ = reshape(g, shape .- Tuple(di))

  res_ .= 0
  @tasks for i ∈ CartesianIndices(g_)
    @inbounds res_[i]  = g_[i]
  end
  @tasks for i ∈ CartesianIndices(g_)
    @inbounds res_[i + di] -= g_[i]
  end
end
