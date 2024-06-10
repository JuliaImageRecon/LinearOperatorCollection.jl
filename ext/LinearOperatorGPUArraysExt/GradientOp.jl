function LinearOperatorCollection.grad!(res::vecT, img::vecT, shape, dim) where {vecT <: AbstractGPUVector}
  δ = zeros(Int, length(shape))
  δ[dim] = 1
  δ = Tuple(δ)
  di = CartesianIndex(δ)

  gpu_call(reshape(res, shape .- δ), reshape(img,shape), di) do ctx, res_, img_, di_
    idx = @cartesianidx(res_)
    @inbounds res_[idx] = img_[idx] - img_[idx + di_]
    return nothing  
  end
  
  return res
end

# adjoint of directional gradients
function LinearOperatorCollection.grad_t!(res::vecT, g::vecT, shape::NTuple{N,Int64}, dim::Int64) where {T, vecT <: AbstractGPUVector{T}, N}
  δ = zeros(Int, length(shape))
  δ[dim] = 1
  δ = Tuple(δ)
  di = CartesianIndex(δ)

  res_ = reshape(res,shape)
  g_ = reshape(g, shape .- δ)

  fill!(res, zero(T))
  gpu_call(res_, g_, di, elements = length(g)) do ctx, res_k, g_k, di_k
    idx = @cartesianidx(g_k)
    @inbounds res_k[idx]  = g_k[idx]
    return nothing  
  end

  gpu_call(res_, g_, di, elements = length(g)) do ctx, res_k, g_k, di_k
    idx = @cartesianidx(g_k)
    @inbounds res_k[idx + di_k] -= g_k[idx]
    return nothing  
  end
end
