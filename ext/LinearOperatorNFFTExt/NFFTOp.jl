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
* (`kargs`)                 - additional keyword arguments for the NFFT plan
* (`cache=true`)            - cache the plan-constructor arguments for
  `Adapt.adapt`; set to `false` to opt out of that reconstruction support

`Adapt.adapt(target, op)` preserves the node geometry, NFFT backend, plan
settings, and `toeplitz`; it replays the cached constructor arguments with
target-backed nodes and rebuilds plans, workspaces, and closures for the
target. If `cache=false`, adapting the operator raises an explicit
`ArgumentError` because the constructor arguments were not retained.
"""
function LinearOperatorCollection.NFFTOp(::Type{T};
    shape::Tuple, nodes::AbstractMatrix{U}, toeplitz=false, oversamplingFactor=1.25, 
   kernelSize=3, precompute = AbstractNFFTs.TENSOR, cache::Bool=true, kargs...) where {U <: Number, T <: Number}
  return NFFTOpImpl(shape, nodes; toeplitz, oversamplingFactor, kernelSize, cache, kargs... )
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
  # Replay the public plan-constructor arguments during Adapt instead of
  # depending on backend-specific plan internals. `nothing` means that the
  # caller explicitly opted out with `cache=false`.
  const plan_args::Union{Nothing,Dict{Symbol,Any}}
end

LinearOperators.storage_type(::NFFTOpImpl{T, vecT}) where {T, vecT} = vecT

function _copy_nfft_plan_value(value)
  value isa AbstractArray || return value
  copied = similar(value, size(value))
  copyto!(copied, value)
  copied
end

function _adapt_nfft_plan_value(to, value)
  value isa AbstractArray || return value
  adapted = Adapt.adapt(to, value)
  adapted isa AbstractArray || throw(ArgumentError(
    "cannot adapt cached NFFT plan argument of type $(typeof(value)) to $(to)"))
  _copy_nfft_plan_value(adapted)
end

function _nfft_plan_cache(backend, tr, shape, plan_kwargs, cache)
  cache || return nothing
  Dict{Symbol,Any}(
    :backend => backend,
    :nodes => _copy_nfft_plan_value(tr),
    :shape => shape,
    :kwargs => Dict{Symbol,Any}(
      key => _copy_nfft_plan_value(value)
      for (key, value) in pairs(plan_kwargs)
    ),
  )
end

function NFFTOpImpl(shape::Tuple, tr::AbstractMatrix{T}; toeplitz, oversamplingFactor, kernelSize,
                    S = Vector{Complex{T}}, cache::Bool=true, kargs...) where {T}

  baseArrayType = Base.typename(S).wrapper # https://github.com/JuliaLang/julia/issues/35543
  backend = AbstractNFFTs.nfft_backend[][]
  plan_kwargs = merge((m=kernelSize, σ=oversamplingFactor,
                       fftflags=FFTW.ESTIMATE, blocking=true), (; kargs...))
  plan = plan_nfft(backend, baseArrayType, tr, shape; plan_kwargs...)
  plan_args = _nfft_plan_cache(backend, tr, shape, plan_kwargs, cache)

  return NFFTOpImpl{eltype(S), S, typeof(plan)}(size(tr,2), prod(shape), false, false
            , (res,x) -> produ!(res,plan,x)
            , nothing
            , (res,y) -> ctprodu!(res,plan,y)
            , 0, 0, 0, S(undef, 0), S(undef, 0)
            , plan, toeplitz, plan_args)
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
  op.plan_args === nothing && throw(ArgumentError(
    "cannot adapt NFFTOp constructed with cache=false: plan arguments were not retained"))
  source_args = op.plan_args
  targetArrayType = Base.typename(typeof(target_storage)).wrapper
  target_nodes = _adapt_nfft_plan_value(to, source_args[:nodes])
  target_kwargs = Dict{Symbol,Any}(
    key => _adapt_nfft_plan_value(to, value)
    for (key, value) in source_args[:kwargs]
  )
  plan = plan_nfft(source_args[:backend], targetArrayType, target_nodes,
                   source_args[:shape]; target_kwargs...)
  S = typeof(target_storage)
  plan_args = Dict{Symbol,Any}(
    :backend => source_args[:backend],
    :nodes => target_nodes,
    :shape => source_args[:shape],
    :kwargs => target_kwargs,
  )
  return NFFTOpImpl{eltype(S), S, typeof(plan)}(
    prod(size_out(plan)), prod(size_in(plan)), false, false,
    (res,x) -> produ!(res, plan, x), nothing,
    (res,y) -> ctprodu!(res, plan, y),
    0, 0, 0, similar(target_storage, 0), similar(target_storage, 0),
    plan, op.toeplitz, plan_args)
end

function produ!(y::AbstractVector, plan::AbstractNFFTPlan, x::AbstractVector) 
  mul!(y, plan, reshape(x,size_in(plan)))
end

function ctprodu!(x::AbstractVector, plan::AbstractNFFTPlan, y::AbstractVector)
  mul!(reshape(x, size_in(plan)), adjoint(plan), y)
end


function Base.copy(S::NFFTOpImpl{T, vecT, P}) where {T, vecT, P}
  plan = copy(S.plan)
  plan_args = if isnothing(S.plan_args)
    nothing
  else
    Dict{Symbol,Any}(
      :backend => S.plan_args[:backend],
      :nodes => _copy_nfft_plan_value(S.plan_args[:nodes]),
      :shape => S.plan_args[:shape],
      :kwargs => copy(S.plan_args[:kwargs]),
    )
  end
  return NFFTOpImpl{T, vecT, P}(size(plan.k,2), prod(size_in(plan)), false, false
              , (res,x) -> produ!(res,plan,x)
              , nothing
              , (res,y) -> ctprodu!(res,plan,y)
              , 0, 0, 0, vecT(undef, 0), vecT(undef, 0)
              , plan, S.toeplitz, plan_args)
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
  # FFT planner kwargs are retained for target reconstruction. `nothing`
  # means that the caller explicitly opted out with `cache=false`.
  const plan_kwargs::Union{Nothing,Dict{Symbol,Any}}
end

LinearOperators.storage_type(::NFFTToeplitzNormalOp{T, vecT}) where {T, vecT} = vecT

"""
    NFFTToeplitzNormalOp(...)

Construct a Toeplitz normal operator for an NFFT. `Adapt.adapt(target, op)`
keeps the shape, weights, Toeplitz kernel `λ`, and FFT planner options, and
rebuilds the FFT plans, workspaces, and closures on the target. Set
`cache=false` when constructing the operator to opt out of retaining planner
kwargs; adapting such an operator raises an explicit `ArgumentError`.
"""
function LinearOperatorCollection.NFFTToeplitzNormalOp(shape, W, fftplan, ifftplan,
                                                       λ, xL1::matT, xL2::matT,
                                                       S = Vector{T}; cache::Bool=true,
                                                       fft_kwargs=NamedTuple()) where {T, D, matT <: AbstractArray{T, D}}

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

  plan_kwargs = cache ? Dict{Symbol,Any}(:fft_kwargs => Dict{Symbol,Any}(pairs(fft_kwargs))) : nothing
  return NFFTToeplitzNormalOp(prod(shape), prod(shape), false, false
         , (res,x) -> produ!(res, shape, fftplan, ifftplan, λ, xL1, xL2, x)
         , nothing
         , nothing
         , 0, 0, 0, S(undef, 0), S(undef, 0)
         , shape, W, fftplan, ifftplan, λ, xL1, xL2, plan_kwargs)
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
  op.plan_kwargs === nothing && throw(ArgumentError(
    "cannot adapt NFFTToeplitzNormalOp constructed with cache=false: " *
    "FFT plan kwargs were not retained"))
  xL1 = similar(adapted_workspace, size(op.xL1))
  xL2 = similar(xL1)
  λ = _adapt_copy(to, op.λ)
  fft_kwargs = op.plan_kwargs[:fft_kwargs]
  fftplan = plan_fft(xL1; fft_kwargs...)
  ifftplan = plan_ifft(xL1; fft_kwargs...)
  weights = if isnothing(op.weights)
    nothing
  elseif op.weights isa AbstractArray
    _adapt_copy(to, op.weights)
  else
    Adapt.adapt(to, op.weights)
  end
  S = typeof(target_storage)
  return LinearOperatorCollection.NFFTToeplitzNormalOp(
    op.shape, weights, fftplan, ifftplan, λ, xL1, xL2, S;
    cache=true, fft_kwargs)
end

# TODO: use vecT for toeplitz op
function LinearOperatorCollection.NFFTToeplitzNormalOp(nfft::NFFTOp{T}, W=nothing;
                                                       S = LinearOperators.storage_type(nfft),
                                                       cache::Bool=true, kwargs...) where {T}
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

  return LinearOperatorCollection.NFFTToeplitzNormalOp(
    shape, W, fftplan, ifftplan, λ, xL1, xL2, S;
    cache, fft_kwargs)
end

function LinearOperatorCollection.normalOperator(S::NFFTOpImpl{T}, W = nothing;
                                                copyOpsFn = copy, cache::Bool=true,
                                                kwargs...) where T
  if S.toeplitz
    return LinearOperatorCollection.NFFTToeplitzNormalOp(S, W; cache, kwargs...)
  else
    return NormalOp(eltype(S); parent = S, weights = W)
  end
end

function Base.copy(A::NFFTToeplitzNormalOp{T,vecT,D,W}) where {T,vecT,D,W}
  fftplan  = plan_fft( zeros(T, 2 .* A.shape); flags=FFTW.MEASURE)
  ifftplan = plan_ifft(zeros(T, 2 .* A.shape); flags=FFTW.MEASURE)
  fft_kwargs = (flags=FFTW.MEASURE,)
  return LinearOperatorCollection.NFFTToeplitzNormalOp(
    A.shape, A.weights, fftplan, ifftplan, A.λ, copy(A.xL1), copy(A.xL2);
    cache=!isnothing(A.plan_kwargs), fft_kwargs)
end
