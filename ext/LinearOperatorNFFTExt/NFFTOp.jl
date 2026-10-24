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

`Adapt.adapt(target, op)` preserves the node geometry, NFFT backend, plan
settings that affect the transform, and `toeplitz`; it rebuilds plans,
workspaces, and closures for the target. Planner-only choices are reset.
"""
function LinearOperatorCollection.NFFTOp(::Type{T};
    shape::Tuple, nodes::AbstractMatrix{U}, toeplitz=false, oversamplingFactor=1.25, 
   kernelSize=3, precompute = AbstractNFFTs.TENSOR, kargs...) where {U <: Number, T <: Number}
  return NFFTOpImpl(shape, nodes; toeplitz, oversamplingFactor, kernelSize, kargs... )
end

mutable struct NFFTOpImpl{T, vecT, P} <: NFFTOp{T, P}
  const nrow :: Int
  const ncol :: Int
  const symmetric :: Bool
  const hermitian :: Bool
  const prod! :: Function
  const tprod! :: Nothing
  const ctprod! :: Function
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  Mv :: vecT
  Mtu :: vecT
  const plan :: P
  toeplitz :: Bool
end

LinearOperators.storage_type(::NFFTOpImpl{T, vecT}) where {T, vecT} = vecT

function _nfft_backend(plan)
  mod = parentmodule(typeof(plan))
  for _ in 1:8
    if isdefined(mod, :backend)
      backend = getfield(mod, :backend)
      if backend isa Function
        candidate = try
          backend()
        catch
          nothing
        end
        candidate isa AbstractNFFTs.AbstractNFFTBackend && return candidate
      end
    end
    parent = parentmodule(mod)
    parent === mod && break
    mod = parent
  end
  throw(ArgumentError("cannot recover the AbstractNFFT backend for plan type $(typeof(plan))"))
end

function _nfft_jl_grid(shape::NTuple{D}, σ) where D
  ntuple(d -> (ceil(Int, σ * shape[d]) ÷ 2) * 2, D)
end

function _nfft_jl_sigma(shape::NTuple{D}, oversampled::NTuple{D}, ::Type{T}) where {D,T <: AbstractFloat}
  lower = maximum(T(oversampled[d] - 1) / T(shape[d]) for d in 1:D)
  upper = minimum(T(oversampled[d] + 1) / T(shape[d]) for d in 1:D)
  lower < upper || throw(ArgumentError(
    "cannot recover an NFFT.jl oversampling factor for N=$(shape), Ñ=$(oversampled)"))

  candidates = (max(one(T), (lower + upper) / 2),
                max(one(T), nextfloat(lower)),
                max(one(T), prevfloat(upper)),
                max(one(T), upper), one(T))
  for σ in candidates
    _nfft_jl_grid(shape, σ) == oversampled && return σ
  end
  throw(ArgumentError(
    "no supported NFFT.jl oversampling factor reproduces N=$(shape), Ñ=$(oversampled)"))
end

function _nextprod_lower(bound::Integer)
  lower = bound
  while lower > 1 && nextprod((2, 3, 5), lower - 1) == bound
    lower -= 1
  end
  lower
end

function _nonuniform_grid(shape::NTuple{D}, σ) where D
  ntuple(d -> nextprod((2, 3, 5), floor(Int, σ * shape[d])), D)
end

function _nonuniform_sigma(shape::NTuple{D}, oversampled::NTuple{D}, ::Type{T}) where {D,T <: AbstractFloat}
  lower = maximum(T(_nextprod_lower(oversampled[d])) / T(shape[d]) for d in 1:D)
  upper = minimum(T(oversampled[d] + 1) / T(shape[d]) for d in 1:D)
  lower < upper || throw(ArgumentError(
    "cannot recover a NonuniformFFTs oversampling factor for N=$(shape), Ñ=$(oversampled)"))

  candidates = (max(one(T), (lower + upper) / 2),
                max(one(T), nextfloat(lower)),
                max(one(T), prevfloat(upper)),
                max(one(T), upper), one(T))
  for σ in candidates
    _nonuniform_grid(shape, σ) == oversampled && return σ
  end
  throw(ArgumentError(
    "no supported NonuniformFFTs oversampling factor reproduces N=$(shape), Ñ=$(oversampled)"))
end

function _recover_nonuniform_kernel(data)
  kernel_module = parentmodule(typeof(data))
  if isdefined(kernel_module, :KaiserBesselKernelData) &&
      data isa getfield(kernel_module, :KaiserBesselKernelData)
    return getfield(kernel_module, :KaiserBesselKernel)(getfield(data, :β))
  elseif isdefined(kernel_module, :BackwardsKaiserBesselKernelData) &&
      data isa getfield(kernel_module, :BackwardsKaiserBesselKernelData)
    return getfield(kernel_module, :BackwardsKaiserBesselKernel)(getfield(data, :β))
  elseif isdefined(kernel_module, :GaussianKernelData) &&
      data isa getfield(kernel_module, :GaussianKernelData)
    gridstep = getfield(kernel_module, :gridstep)
    return getfield(kernel_module, :GaussianKernel)(getfield(data, :σ) / gridstep(data))
  elseif isdefined(kernel_module, :BSplineKernelData) &&
      data isa getfield(kernel_module, :BSplineKernelData)
    return getfield(kernel_module, :BSplineKernel)()
  end
  throw(ArgumentError(
    "cannot recover the NonuniformFFTs kernel from plan data $(typeof(data))"))
end

function _recover_point_transform(plan)
  fold = getproperty(plan, :point_transform_fold)
  hasfield(typeof(fold), :point_transform) || throw(ArgumentError(
    "cannot recover NonuniformFFTs point_transform from $(typeof(fold))"))
  getfield(fold, :point_transform)
end

function NFFTOpImpl(shape::Tuple, tr::AbstractMatrix{T}; toeplitz, oversamplingFactor, kernelSize, S = Vector{Complex{T}}, kargs...) where {T}

  baseArrayType = Base.typename(S).wrapper # https://github.com/JuliaLang/julia/issues/35543
  backend = AbstractNFFTs.nfft_backend[][]
  plan = plan_nfft(backend, baseArrayType, tr, shape; m=kernelSize, σ=oversamplingFactor,
		                          fftflags=FFTW.ESTIMATE, blocking=true, kargs...)

  return NFFTOpImpl{eltype(S), S, typeof(plan)}(size(tr,2), prod(shape), false, false
            , (res,x) -> produ!(res,plan,x)
            , nothing
            , (res,y) -> ctprodu!(res,plan,y)
            , 0, 0, 0, S(undef, 0), S(undef, 0)
            , plan, toeplitz)
end

function _rebuild_nfft_jl_plan(to, plan, targetArrayType)
  shape = Tuple(plan.N)
  oversampled = Tuple(plan.Ñ)
  σ = _nfft_jl_sigma(shape, oversampled, eltype(plan.k))
  rebuilt = plan_nfft(_nfft_backend(plan), targetArrayType, copy(plan.k), shape;
    m=plan.params.m, σ, window=plan.params.window,
    fftflags=FFTW.ESTIMATE, blocking=true)
  getproperty(rebuilt, :Ñ) == plan.Ñ || throw(ArgumentError(
    "NFFT.jl target plan changed oversampled grid from $(plan.Ñ) to $(rebuilt.Ñ)"))
  rebuilt
end

function _rebuild_nonuniform_plan(to, plan, targetArrayType)
  inner = getproperty(plan, :p)
  shape = Tuple(size_in(plan))
  oversampled = Tuple(size(first(getproperty(inner, :data).us)))
  points = getproperty(inner, :points)
  nodes = Array(stack(points; dims=1))
  data = first(getproperty(inner, :kernels))
  kernel_module = parentmodule(typeof(data))
  half_support = getfield(kernel_module, :half_support)(data)
  kernel = _recover_nonuniform_kernel(data)
  σ = _nonuniform_sigma(shape, oversampled, eltype(first(points)))
  point_transform = try
    Adapt.adapt(to, _recover_point_transform(inner))
  catch err
    throw(ArgumentError("cannot adapt NonuniformFFTs point_transform for $(targetArrayType): $(sprint(showerror, err))"))
  end
  rebuilt = plan_nfft(_nfft_backend(plan), targetArrayType, nodes, shape;
    m=half_support, σ, window=kernel, fftshift=inner.fftshift,
    point_transform, kernel_evalmode=inner.kernel_evalmode,
    fftflags=FFTW.ESTIMATE)
  rebuilt_shape = Tuple(size(first(getproperty(rebuilt.p, :data).us)))
  rebuilt_shape == oversampled || throw(ArgumentError(
    "NonuniformFFTs target plan changed oversampled grid from $(oversampled) to $(rebuilt_shape)"))
  rebuilt
end

function _rebuild_nfft_plan(to, plan::AbstractNFFTs.AbstractNFFTPlan, targetArrayType)
  if hasproperty(plan, :N) && hasproperty(plan, :Ñ) &&
      hasproperty(plan, :params) && hasproperty(plan, :k)
    _rebuild_nfft_jl_plan(to, plan, targetArrayType)
  elseif hasproperty(plan, :p) && hasproperty(getproperty(plan, :p), :kernels)
    _rebuild_nonuniform_plan(to, plan, targetArrayType)
  else
    throw(ArgumentError("unsupported NFFT plan type $(typeof(plan)) for Adapt reconstruction"))
  end
end

function _adapt_target_storage(to, storage, operator_name)
  target_storage = Adapt.adapt(to, storage)
  supported = to isa Type ? (!isabstracttype(to) && typeof(target_storage) <: to) :
                             target_storage !== storage
  supported && target_storage isa AbstractArray || throw(ArgumentError(
    "unsupported Adapt target $(to) for $(operator_name): " *
    "Adapt.adapt returned $(typeof(target_storage))"))
  target_storage
end

function Adapt.adapt_structure(to, op::NFFTOpImpl)
  target_storage = _adapt_target_storage(to, op.Mv, "NFFTOp")
  targetArrayType = Base.typename(typeof(target_storage)).wrapper
  plan = _rebuild_nfft_plan(to, op.plan, targetArrayType)
  S = typeof(target_storage)
  return NFFTOpImpl{eltype(S), S, typeof(plan)}(
    prod(size_out(plan)), prod(size_in(plan)), false, false,
    (res,x) -> produ!(res, plan, x), nothing,
    (res,y) -> ctprodu!(res, plan, y),
    0, 0, 0, similar(target_storage, 0), similar(target_storage, 0),
    plan, op.toeplitz)
end

function produ!(y::AbstractVector, plan::AbstractNFFTPlan, x::AbstractVector) 
  mul!(y, plan, reshape(x,size_in(plan)))
end

function ctprodu!(x::AbstractVector, plan::AbstractNFFTPlan, y::AbstractVector)
  mul!(reshape(x, size_in(plan)), adjoint(plan), y)
end


function Base.copy(S::NFFTOpImpl{T, vecT, P}) where {T, vecT, P}
  plan = copy(S.plan)
  return NFFTOpImpl{T, vecT, P}(size(plan.k,2), prod(size_in(plan)), false, false
              , (res,x) -> produ!(res,plan,x)
              , nothing
              , (res,y) -> ctprodu!(res,plan,y)
              , 0, 0, 0, vecT(undef, 0), vecT(undef, 0)
              , plan, S.toeplitz)
end



#########################################################################
### Toeplitz Operator ###
#########################################################################

mutable struct NFFTToeplitzNormalOp{T, vecT <: AbstractVector{T}, D,W, matT <: AbstractArray{T, D}, P <: AbstractFFTs.Plan, IP <: AbstractFFTs.Plan} <: AbstractLinearOperator{T}
  const nrow :: Int
  const ncol :: Int
  const symmetric :: Bool
  const hermitian :: Bool
  const prod! :: Function
  const tprod! :: Nothing
  const ctprod! :: Nothing
  nprod :: Int
  ntprod :: Int
  nctprod :: Int
  Mv :: vecT
  Mtu :: vecT
  shape::NTuple{D,Int}
  weights::W
  fftplan :: P
  ifftplan :: IP
  λ::matT
  xL1::matT
  xL2::matT
end

LinearOperators.storage_type(::NFFTToeplitzNormalOp{T, vecT}) where {T, vecT} = vecT

"""
    NFFTToeplitzNormalOp(...)

Construct a Toeplitz normal operator for an NFFT. `Adapt.adapt(target, op)`
keeps the shape, weights, and Toeplitz kernel `λ`, and rebuilds the FFT plans,
workspaces, and closures on the target. FFT planner options are reset.
"""
function LinearOperatorCollection.NFFTToeplitzNormalOp(shape, W, fftplan, ifftplan, λ, xL1::matT, xL2::matT, S = Vector{T}) where {T, D, matT <: AbstractArray{T, D}}

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
         , 0, 0, 0, S(undef, 0), S(undef, 0)
         , shape, W, fftplan, ifftplan, λ, xL1, xL2)
end

function _adapt_copy(to, value)
  adapted = Adapt.adapt(to, value)
  if adapted isa AbstractArray
    copied = similar(adapted)
    copyto!(copied, adapted)
    return copied
  end
  adapted
end

function Adapt.adapt_structure(to, op::NFFTToeplitzNormalOp)
  target_storage = _adapt_target_storage(to, op.Mv, "NFFTToeplitzNormalOp")
  adapted_workspace = _adapt_target_storage(to, op.xL1, "NFFTToeplitzNormalOp")
  xL1 = similar(adapted_workspace, size(op.xL1))
  xL2 = similar(xL1)
  λ = _adapt_copy(to, op.λ)
  fftplan = plan_fft(xL1; flags=FFTW.ESTIMATE)
  ifftplan = plan_ifft(xL1; flags=FFTW.ESTIMATE)
  weights = if isnothing(op.weights)
    nothing
  elseif op.weights isa AbstractArray
    _adapt_copy(to, op.weights)
  else
    Adapt.adapt(to, op.weights)
  end
  S = typeof(target_storage)
  return LinearOperatorCollection.NFFTToeplitzNormalOp(
    op.shape, weights, fftplan, ifftplan, λ, xL1, xL2, S)
end

# TODO: use vecT for toeplitz op
function LinearOperatorCollection.NFFTToeplitzNormalOp(nfft::NFFTOp{T}, W=nothing; S = LinearOperators.storage_type(nfft), kwargs...) where {T}
  shape = size_in(nfft.plan)

  tmpVec = similar(nfft.Mv, (2 .* shape)...)
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

  λ = similar(tmpVec)
  λ .= fftplan * fftshift(eigMat)

  xL1 = tmpVec
  xL2 = similar(xL1)

  return LinearOperatorCollection.NFFTToeplitzNormalOp(shape, W, fftplan, ifftplan, λ, xL1, xL2, S)
end

function LinearOperatorCollection.normalOperator(S::NFFTOpImpl{T}, W = nothing; copyOpsFn = copy, kwargs...) where T
  if S.toeplitz
    return LinearOperatorCollection.NFFTToeplitzNormalOp(S,W; kwargs...)
  else
    return NormalOp(eltype(S); parent = S, weights = W)
  end
end

function Base.copy(A::NFFTToeplitzNormalOp{T,vecT,D,W}) where {T,vecT,D,W}
  fftplan  = plan_fft( zeros(T, 2 .* A.shape); flags=FFTW.MEASURE)
  ifftplan = plan_ifft(zeros(T, 2 .* A.shape); flags=FFTW.MEASURE)
  return LinearOperatorCollection.NFFTToeplitzNormalOp(A.shape, A.weights, fftplan, ifftplan, A.λ, copy(A.xL1), copy(A.xL2))
end
