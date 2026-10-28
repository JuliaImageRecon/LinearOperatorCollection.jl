module LinearOperatorNonuniformFFTsExt

using LinearOperatorCollection, AbstractNFFTs, NonuniformFFTs, NonuniformFFTs.Kernels, FFTW
using Adapt

function LinearOperatorCollection.NFFTToeplitzNormalOp(nfft::NFFTOp{T, P}, W=nothing;
                                                       S = LinearOperators.storage_type(nfft),
                                                       cache::Bool=true, kwargs...) where {T, P <: NonuniformFFTs.NFFTPlan}
  shape = size_in(nfft.plan)

  tmpVec = similar(nfft.Mv, (2 .* shape)...)
  tmpVec .= zero(T)

  # plan the FFTs
  fft_kwargs = (; kwargs...)
  fftplan  = plan_fft(tmpVec; fft_kwargs...)
  ifftplan = plan_ifft(tmpVec; fft_kwargs...)

  # TODO extend the following function by weights
  # λ = calculateToeplitzKernel(shape, nfft.plan.k; m = nfft.plan.params.m, σ = nfft.plan.params.σ, window = nfft.plan.params.window, LUTSize = nfft.plan.params.LUTSize, fftplan = fftplan)

  shape_os = 2 .* shape
  baseArrayType = Base.typename(typeof(tmpVec)).wrapper # https://github.com/JuliaLang/julia/issues/35543

  nufft = nfft.plan.p
  σ = nufft.σ
  m = Kernels.half_support(first(nufft.kernels))
  k = stack(nufft.points, dims = 1)

  p = plan_nfft(baseArrayType, k, shape_os; m = m, σ = σ,
		precompute=AbstractNFFTs.POLYNOMIAL, fftflags=FFTW.ESTIMATE, blocking=true)
  tmpOnes = similar(tmpVec, size(k, 2))
  tmpOnes .= one(T)
  
  if !isnothing(W)
    eigMat = adjoint(p) * ( W  * tmpOnes)
  else
    eigMat = adjoint(p) * (tmpOnes)
  end

  λ = similar(tmpVec)
  λ .= fftplan * fftshift(eigMat)

  xL1 = tmpVec
  xL2 = similar(xL1)

  return LinearOperatorCollection.NFFTToeplitzNormalOp(
    shape, W, fftplan, ifftplan, λ, xL1, xL2, S;
    cache, fft_kwargs)
end


end
