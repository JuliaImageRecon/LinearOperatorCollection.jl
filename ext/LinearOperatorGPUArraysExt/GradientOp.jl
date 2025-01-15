function LinearOperatorCollection.grad!(res::vecT, img::vecT, shape::NTuple{N,Int64}, di::CartesianIndex{N}) where {vecT <: AbstractGPUVector, N}
  res = reshape(res, shape .- Tuple(di))
  backend = get_backend(res)

  @kernel cpu = false inbounds = true function grad_kernel!(res, img, di)
    idx = @index(Global, Cartesian)
    res[idx] = img[idx] - img[idx + di]
  end

  if length(res) > 0
    kernel = grad_kernel!(backend)
    kernel(res, reshape(img, shape), di, ndrange = size(res))
  end

  return res
end


# adjoint of directional gradients
function LinearOperatorCollection.grad_t!(res::vecT, g::vecT, shape::NTuple{N,Int64}, di::CartesianIndex{N}) where {T, vecT <: AbstractGPUVector{T}, N}
  res_ = reshape(res,shape)
  g_ = reshape(g, shape .- Tuple(di))
  backend = get_backend(res)

  fill!(res, zero(T))
  if length(g_) > 0
    kernel1 = grad_t_kernel_1!(backend)
    kernel2 = grad_t_kernel_2!(backend)
    kernel1(res_, g_, di, ndrange = size(g_))
    kernel2(res_, g_, di, ndrange = size(g_))
  end
  
  return res
end

@kernel cpu = false inbounds = true function grad_t_kernel_1!(res, g, di)
  idx = @index(Global, Cartesian)
  res[idx] += g[idx]  
end

@kernel cpu = false inbounds = true function grad_t_kernel_2!(res, g, di)
  idx = @index(Global, Cartesian)
  res[idx + di] -= g[idx]  
end


function LinearOperatorCollection.grad_t!(res::vecT, g::vecT, shape::NTuple{N,Int64}, dirs, dims, dim_ends, tmp) where {T, vecT <: AbstractGPUVector{T}, N}
  dim_start = 1
  res = reshape(res, shape)
  backend = get_backend(res)

  fill!(res, zero(eltype(res)))
  kernel1 = grad_t_kernel_1!(backend)
  kernel2 = grad_t_kernel_2!(backend)
  for (i, di) in enumerate(dirs)
    g_ = reshape(view(g, dim_start:dim_ends[i]), shape .- Tuple(di))
    if length(g_) > 0
      kernel1(res, g_, di, ndrange = size(g_))
      kernel2(res, g_, di, ndrange = size(g_))
    end  
    dim_start = dim_ends[i] + 1
  end
  return res
end