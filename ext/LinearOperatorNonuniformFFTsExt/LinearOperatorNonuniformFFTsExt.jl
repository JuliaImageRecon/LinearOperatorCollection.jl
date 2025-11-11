module LinearOperatorNonuniformFFTsExt

using LinearOperatorCollection, AbstractNFFTs, NonuniformFFTs, NonuniformFFTs.Kernels, FFTW

function LinearOperatorCollection.NFFTToeplitzNormalOp(nfft::NFFTOp{T, P}, W=nothing; kwargs...) where {T, vecT, P <: NonuniformFFTs.NFFTPlan}
  shape = size_in(nfft.plan)

  tmpVec = similar(nfft.Mv5, (2 .* shape)...)
  tmpVec .= zero(T)

  # plan the FFTs
  fftplan  = plan_fft(tmpVec; kwargs...)
  ifftplan = plan_ifft(tmpVec; kwargs...)

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

  λ = fftplan * fftshift(eigMat)

  xL1 = tmpVec
  xL2 = similar(xL1)

  return LinearOperatorCollection.NFFTToeplitzNormalOp(shape, W, fftplan, ifftplan, λ, xL1, xL2)
end


end