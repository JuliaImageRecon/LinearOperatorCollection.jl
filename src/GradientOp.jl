"""
  GradientOp(T::Type; shape::Tuple, dims=1:length(shape))

directional gradient operator along the dimensions `dims` for an array of size `shape`.

# Required Argument
  * `T`                        - type of elements, .e.g. `Float64` for `ComplexF32`

# Required Keyword argument
  * `shape::NTuple{N,Int}`     - shape of the array (e.g., image)

# Optional Keyword argument
  * `dims`                     - dimension(s) along which the gradient is applied; default is `1:length(shape)`
"""
function GradientOp(::Type{T}; shape::NTuple{N,Int}, dims=1:length(shape), kwargs...) where {T <: Number, N}
  return GradientOpImpl(T, shape, dims; kwargs...)
end

function GradientOpImpl(T::Type, shape::NTuple{N,Int}, dims; S = Vector{T}) where N
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
  
  return LinearOperator{T}(nrow, ncol, false, false,
                          (res,x) -> (grad!(res,x,shape,dirs, dims, dim_ends)),
                          (res,x) -> (grad_t!(res,x,shape,dirs, dims, dim_ends, tmp)),
                          (res,x) -> (grad_t!(res,x,shape,dirs, dims, dim_ends, tmp)),
                          S = S)
end

function GradientOpImpl(T::Type, shape::NTuple{N,Int}, dim::Int; S = Vector{T}) where N
  nrow = div( (shape[dim]-1)*prod(shape), shape[dim] )
  ncol = prod(shape)
  δ = zeros(Int, length(shape))
  δ[dim] = 1
  δ = Tuple(δ)
  dir = CartesianIndex(δ)
  return LinearOperator{T}(nrow, ncol, false, false,
                          (res,x) -> (grad!(res,x,shape,dir)),
                          (res,x) -> (grad_t!(res,x,shape,dir)),
                          (res,x) -> (grad_t!(res,x,shape,dir)),
                          S = S)
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

  Threads.@threads for i ∈ CartesianIndices(res_)
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
  Threads.@threads for i ∈ CartesianIndices(g_)
    @inbounds res_[i]  = g_[i]
  end
  Threads.@threads for i ∈ CartesianIndices(g_)
    @inbounds res_[i + di] -= g_[i]
  end
end
