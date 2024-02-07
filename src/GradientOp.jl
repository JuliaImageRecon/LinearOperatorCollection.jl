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
function GradientOp(::Type{T}; shape::NTuple{N,Int}, dims=1:length(shape)) where {T <: Number, N}
  return GradientOpImpl(T, shape, dims)
end

function GradientOpImpl(T::Type, shape::NTuple{N,Int}, dims) where N
  return vcat([GradientOpImpl(T, shape, dim) for dim ∈ dims]...)
end

function GradientOpImpl(T::Type, shape::NTuple{N,Int}, dim::Int) where N
  nrow = div( (shape[dim]-1)*prod(shape), shape[dim] )
  ncol = prod(shape)
  return LinearOperator{T}(nrow, ncol, false, false,
                          (res,x) -> (grad!(res,x,shape,dim)),
                          (res,x) -> (grad_t!(res,x,shape,dim)),
                          (res,x) -> (grad_t!(res,x,shape,dim))
                          )
end

# directional gradients
function grad!(res::T, img::U, shape, dim) where {T<:AbstractVector, U<:AbstractVector}
  img_ = reshape(img,shape)

  δ = zeros(Int, length(shape))
  δ[dim] = 1
  δ = Tuple(δ)
  di = CartesianIndex(δ)

  res_ = reshape(res, shape .- δ)

  Threads.@threads for i ∈ CartesianIndices(res_)
    @inbounds res_[i] = img_[i] - img_[i + di]
  end
end


# adjoint of directional gradients
function grad_t!(res::T, g::U, shape::NTuple{N,Int64}, dim::Int64) where {T<:AbstractVector, U<:AbstractVector, N}
  δ = zeros(Int, length(shape))
  δ[dim] = 1
  δ = Tuple(δ)
  di = CartesianIndex(δ)

  res_ = reshape(res,shape)
  g_ = reshape(g, shape .- δ)

  res_ .= 0
  Threads.@threads for i ∈ CartesianIndices(g_)
    @inbounds res_[i]  = g_[i]
  end
  Threads.@threads for i ∈ CartesianIndices(g_)
    @inbounds res_[i + di] -= g_[i]
  end
end
