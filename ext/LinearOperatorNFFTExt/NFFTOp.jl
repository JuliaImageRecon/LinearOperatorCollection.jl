
function LinearOperatorCollection.NFFTOp(::Type{T};
    shape::Tuple, nodes::AbstractMatrix{U}, toeplitz=false, oversamplingFactor=1.25, 
   kernelSize=3, kargs...) where {U <: Number, T <: Number}
  return NFFTOpImpl(shape, nodes; toeplitz, oversamplingFactor, kernelSize, kargs... )
end

mutable struct NFFTOpImpl{T, vecT, P <: AbstractNFFTPlan} <: NFFTOp{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Nothing
  ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  Mv5 :: vecT
  Mtu5 :: vecT
  plan :: P
  toeplitz :: Bool
end

LinearOperators.storage_type(op::NFFTOpImpl) = typeof(op.Mv5)

"""
    NFFTOpImpl(shape::Tuple, tr::Trajectory; kargs...)
    NFFTOpImpl(shape::Tuple, tr::AbstractMatrix; kargs...)

generates a `NFFTOpImpl` which evaluates the MRI Fourier signal encoding operator using the NFFT.

# Arguments:
* `shape::NTuple{D,Int64}`  - size of image to encode/reconstruct
* `tr`                      - Either a `Trajectory` object, or a `ND x Nsamples` matrix for an ND-dimenensional (e.g. 2D or 3D) NFFT with `Nsamples` k-space samples
* (`nodes=nothing`)         - Array containg the trajectory nodes (redundant)
* (`kargs`)                 - additional keyword arguments
"""
function NFFTOpImpl(shape::Tuple, tr::AbstractMatrix{T}; toeplitz=false, oversamplingFactor=1.25, kernelSize=3, S = Vector{Complex{T}}, kargs...) where {T}

  plan = plan_nfft(S, tr, shape, m=kernelSize, σ=oversamplingFactor, precompute=NFFT.TENSOR,
		                          fftflags=FFTW.ESTIMATE, blocking=true)

  return NFFTOpImpl{eltype(S), S, typeof(plan)}(size(tr,2), prod(shape), false, false
            , (res,x) -> produ!(res,plan,x)
            , nothing
            , (res,y) -> ctprodu!(res,plan,y)
            , 0, 0, 0, false, false, false, S(), S()
            , plan, toeplitz)
end

function produ!(y::AbstractVector, plan::AbstractNFFTPlan, x::AbstractVector) 
  mul!(y, plan, reshape(x,plan.N))
end

function ctprodu!(x::AbstractVector, plan::AbstractNFFTPlan, y::AbstractVector)
  mul!(reshape(x, plan.N), adjoint(plan), y)
end


function Base.copy(S::NFFTOpImpl{T, vecT, P}) where {T, vecT, P}
  plan = copy(S.plan)
  return NFFTOpImpl{T, vecT, P}(size(plan.k,2), prod(plan.N), false, false
              , (res,x) -> produ!(res,plan,x)
              , nothing
              , (res,y) -> ctprodu!(res,plan,y)
              , 0, 0, 0, false, false, false, vecT(undef, 0), vecT(undef, 0)
              , plan, S.toeplitz)
end



#########################################################################
### Toeplitz Operator ###
#########################################################################

mutable struct NFFTToeplitzNormalOp{T,D,W, vecT <: AbstractVector{T}, matT <: AbstractArray{T, D}, P <: AbstractFFTs.Plan, IP <: AbstractFFTs.Plan} <: AbstractLinearOperator{T}
  nrow :: Int
  ncol :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod! :: Function
  tprod! :: Nothing
  ctprod! :: Nothing
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  args5 :: Bool
  use_prod5! :: Bool
  allocated5 :: Bool
  Mv5 :: vecT
  Mtu5 :: vecT
  shape::NTuple{D,Int}
  weights::W
  fftplan :: P
  ifftplan :: IP
  λ::matT
  xL1::matT
  xL2::matT
end

LinearOperators.storage_type(op::NFFTToeplitzNormalOp) = typeof(op.Mv5)

function NFFTToeplitzNormalOp(shape, W, fftplan, ifftplan, λ, xL1::matT, xL2::matT) where {T, D, matT <: AbstractArray{T, D}}

  function produ!(y, shape, fftplan, ifftplan, λ, xL1, xL2, x)
    xL1 .= 0
    x = reshape(x, shape)
  
    xL1[CartesianIndices(x)] .= x
    mul!(xL2, fftplan, xL1)
    xL2 .*= λ
    mul!(xL1, ifftplan, xL2)
  
    y .= vec(xL1[CartesianIndices(x)])
    return y
  end

  return NFFTToeplitzNormalOp(prod(shape), prod(shape), false, false
         , (res,x) -> produ!(res, shape, fftplan, ifftplan, λ, xL1, xL2, x)
         , nothing
         , nothing
         , 0, 0, 0, false, false, false, T[], T[]
         , shape, W, fftplan, ifftplan, λ, xL1, xL2)
end

function NFFTToeplitzNormalOp(nfft::NFFTOp{T}, W=opEye(eltype(nfft), size(nfft, 1), S= LinearOperators.storage_type(nfft))) where {T}
  shape = nfft.plan.N

  tmpVec = similar(nfft.Mv5, (2 .* shape)...)
  tmpVec .= zero(T)

  # plan the FFTs
  fftplan  = plan_fft(tmpVec; kwargs...)
  ifftplan = plan_ifft(tmpVec; kwargs...)

  # TODO extend the following function by weights
  # λ = calculateToeplitzKernel(shape, nfft.plan.k; m = nfft.plan.params.m, σ = nfft.plan.params.σ, window = nfft.plan.params.window, LUTSize = nfft.plan.params.LUTSize, fftplan = fftplan)

  shape_os = 2 .* shape
  p = plan_nfft(typeof(tmpVec), nfft.plan.k, shape_os; m = nfft.plan.params.m, σ = nfft.plan.params.σ,
		precompute=NFFT.POLYNOMIAL, fftflags=FFTW.ESTIMATE, blocking=true)
  tmpOnes = similar(tmpVec, size(nfft.plan.k, 2))
  tmpOnes .= one(T)
  eigMat = adjoint(p) * ( W  * tmpOnes)
  λ = fftplan * fftshift(eigMat)

  xL1 = tmpVec
  xL2 = similar(xL1)

  return NFFTToeplitzNormalOp(shape, W, fftplan, ifftplan, λ, xL1, xL2)
end

function LinearOperatorCollection.normalOperator(S::NFFTOpImpl{T}, W = opEye(eltype(S), size(S, 1), S= LinearOperators.storage_type(S)), kwargs...) where T
  if S.toeplitz
    return NFFTToeplitzNormalOp(S,W, kwargs...)
  else
    return NormalOp(eltype(S); parent = S, weights = W)
  end
end

function Base.copy(A::NFFTToeplitzNormalOp{T,D,W}) where {T,D,W}
  fftplan  = plan_fft( zeros(T, 2 .* A.shape); flags=FFTW.MEASURE)
  ifftplan = plan_ifft(zeros(T, 2 .* A.shape); flags=FFTW.MEASURE)
  return NFFTToeplitzNormalOp(A.shape, A.weights, fftplan, ifftplan, A.λ, copy(A.xL1), copy(A.xL2))
end
