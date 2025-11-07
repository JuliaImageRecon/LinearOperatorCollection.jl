"""
    NFFTOpImpl(shape::Tuple, tr::Trajectory; kargs...)
    NFFTOpImpl(shape::Tuple, tr::AbstractMatrix; kargs...)

generates a `NFFTOpImpl` which evaluates the MRI Fourier signal encoding operator using the NFFT.

# Arguments:
* `shape::NTuple{D,Int64}`  - size of image to encode/reconstruct
* `nodes=nothing`         - Array containg the trajectory nodes
* `toeplitz=false`        - 
* `oversamplingFactor=1.25`
* `kernelSize=3`
* `precompute = AbstractNFFTs.TENSOR` Precompute flag for the NFFT backend
* (`kargs`)                 - additional keyword arguments for the NFFT plan, 
"""
function LinearOperatorCollection.NFFTOp(::Type{T};
    shape::Tuple, nodes::AbstractMatrix{U}, toeplitz=false, oversamplingFactor=1.25, 
   kernelSize=3, precompute = AbstractNFFTs.TENSOR, kargs...) where {U <: Number, T <: Number}
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

function NFFTOpImpl(shape::Tuple, tr::AbstractMatrix{T}; toeplitz, oversamplingFactor, kernelSize, S = Vector{Complex{T}}, kargs...) where {T}

  baseArrayType = Base.typename(S).wrapper # https://github.com/JuliaLang/julia/issues/35543
  plan = plan_nfft(baseArrayType, tr, shape; m=kernelSize, σ=oversamplingFactor,
		                          fftflags=FFTW.ESTIMATE, blocking=true, kargs...)

  return NFFTOpImpl{eltype(S), S, typeof(plan)}(size(tr,2), prod(shape), false, false
            , (res,x) -> produ!(res,plan,x)
            , nothing
            , (res,y) -> ctprodu!(res,plan,y)
            , 0, 0, 0, false, false, false, S(undef, 0), S(undef, 0)
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

function NFFTToeplitzNormalOp(nfft::NFFTOp{T}, W=nothing; kwargs...) where {T}
  shape = nfft.plan.N

  tmpVec = similar(nfft.Mv5, (2 .* shape)...)
  tmpVec .= zero(T)

  # plan the FFTs
  fftplan  = plan_fft(tmpVec; kwargs...)
  ifftplan = plan_ifft(tmpVec; kwargs...)

  # TODO extend the following function by weights
  # λ = calculateToeplitzKernel(shape, nfft.plan.k; m = nfft.plan.params.m, σ = nfft.plan.params.σ, window = nfft.plan.params.window, LUTSize = nfft.plan.params.LUTSize, fftplan = fftplan)

  shape_os = 2 .* shape
  baseArrayType = Base.typename(typeof(tmpVec)).wrapper # https://github.com/JuliaLang/julia/issues/35543
  p = plan_nfft(baseArrayType, nfft.plan.k, shape_os; m = nfft.plan.params.m, σ = nfft.plan.params.σ,
		precompute=AbstractNFFTs.POLYNOMIAL, fftflags=FFTW.ESTIMATE, blocking=true)
  tmpOnes = similar(tmpVec, size(nfft.plan.k, 2))
  tmpOnes .= one(T)
  
  if !isnothing(W)
    eigMat = adjoint(p) * ( W  * tmpOnes)
  else
    eigMat = adjoint(p) * (tmpOnes)
  end

  λ = fftplan * fftshift(eigMat)

  xL1 = tmpVec
  xL2 = similar(xL1)

  return NFFTToeplitzNormalOp(shape, W, fftplan, ifftplan, λ, xL1, xL2)
end

function LinearOperatorCollection.normalOperator(S::NFFTOpImpl{T}, W = nothing; copyOpsFn = copy, kwargs...) where T
  if S.toeplitz
    return NFFTToeplitzNormalOp(S,W; kwargs...)
  else
    return NormalOp(eltype(S); parent = S, weights = W)
  end
end

function Base.copy(A::NFFTToeplitzNormalOp{T,D,W}) where {T,D,W}
  fftplan  = plan_fft( zeros(T, 2 .* A.shape); flags=FFTW.MEASURE)
  ifftplan = plan_ifft(zeros(T, 2 .* A.shape); flags=FFTW.MEASURE)
  return NFFTToeplitzNormalOp(A.shape, A.weights, fftplan, ifftplan, A.λ, copy(A.xL1), copy(A.xL2))
end
