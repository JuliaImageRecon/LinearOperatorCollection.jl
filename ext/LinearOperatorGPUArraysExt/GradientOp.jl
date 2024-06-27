function LinearOperatorCollection.grad!(res::vecT, img::vecT, shape::NTuple{N,Int64}, di::CartesianIndex{N}) where {vecT <: AbstractGPUVector, N}
  res = reshape(res, shape .- Tuple(di))

  if length(res) > 0
    gpu_call(grad_kernel!, res, reshape(img,shape), di)
  end

  return res
end

function grad_kernel!(ctx, res, img, di)
  idx = @cartesianidx(res)
  @inbounds res[idx] = img[idx] - img[idx + di]
  return nothing  
end

# adjoint of directional gradients
function LinearOperatorCollection.grad_t!(res::vecT, g::vecT, shape::NTuple{N,Int64}, di::CartesianIndex{N}) where {T, vecT <: AbstractGPUVector{T}, N}
  res_ = reshape(res,shape)
  g_ = reshape(g, shape .- Tuple(di))

  fill!(res, zero(T))
  if length(g_) > 0
    gpu_call(grad_t_kernel_1!, res_, g_, di, elements = length(g))
    gpu_call(grad_t_kernel_2!, res_, g_, di, elements = length(g))
  end
end

function grad_t_kernel_1!(ctx, res, g, di)
  idx = @cartesianidx(g)
  @inbounds res[idx] += g[idx]
  return nothing  
end

function grad_t_kernel_2!(ctx, res, g, di)
  idx = @cartesianidx(g)
  @inbounds res[idx + di] -= g[idx]
  return nothing  
end

function LinearOperatorCollection.grad_t!(res::vecT, g::vecT, shape::NTuple{N,Int64}, dirs, dims, dim_ends, tmp) where {T, vecT <: AbstractGPUVector{T}, N}
  dim_start = 1
  res = reshape(res, shape)

  fill!(res, zero(eltype(res)))
  for (i, di) in enumerate(dirs)
    g_ = reshape(view(g, dim_start:dim_ends[i]), shape .- Tuple(di))
    if length(g_) > 0
      gpu_call(grad_t_kernel_1!, res, g_, di, elements = length(g))
      gpu_call(grad_t_kernel_2!, res, g_, di, elements = length(g))
    end  
    dim_start = dim_ends[i] + 1
  end
end