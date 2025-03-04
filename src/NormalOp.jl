export normalOperator

"""
    NormalOp(T::Type; parent, weights)
  
Lazy normal operator of `parent` with an optional weighting operator `weights.`
Computes `adjoint(parent) * weights * parent`.

# Required Argument
  * `T`                        - type of elements, .e.g. `Float64` for `ComplexF32`

# Required Keyword argument
  * `parent`                   - Base operator

# Optional Keyword argument
  * `weights`                  - Optional weights for normal operator. Must already be of form `weights = adjoint.(w) .* w`

"""
function LinearOperatorCollection.NormalOp(::Type{T}; parent, weights = nothing) where T <: Number
  return NormalOp(T, parent, weights)
end

NormalOp(::Union{Type{T}, Type{Complex{T}}}, parent, weights::AbstractVector{T}) where T = NormalOp(T, parent, WeightingOp(weights))

NormalOp(::Union{Type{T}, Type{Complex{T}}}, parent, weights; kwargs...) where T = NormalOpImpl(parent, weights)

mutable struct NormalOpImpl{T,S,D,V} <: NormalOp{T, S}
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
  Mv5 :: V
  Mtu5 :: V
  parent::S
  weights::D
  tmp::V
end

LinearOperators.storage_type(op::NormalOpImpl) = typeof(op.Mv5)

function NormalOpImpl(parent, weights)
  S = promote_storage_types(parent, weights)
  tmp = S(undef, size(parent, 1))
  return NormalOpImpl(parent, weights, tmp)
end
function NormalOpImpl(parent, weights::Nothing)
  S = storage_type(parent)
  tmp = S(undef, size(parent, 1))
  return NormalOpImpl(parent, weights, tmp)
end

function NormalOpImpl(parent, weights, tmp)
  function produ!(y, parent, weights, tmp, x)
    mul!(tmp, parent, x)
    mul!(tmp, weights, tmp) # This can be dangerous. We might need to create two tmp vectors
    return mul!(y, adjoint(parent), tmp)
  end
  function produ!(y, parent, weights::Nothing, tmp, x)
    mul!(tmp, parent, x)
    return mul!(y, adjoint(parent), tmp)
  end


  return NormalOpImpl{eltype(parent), typeof(parent), typeof(weights), typeof(tmp)}(size(parent,2), size(parent,2), false, false
         , (res,x) -> produ!(res, parent, weights, tmp, x)
         , nothing
         , nothing
         , 0, 0, 0, true, false, true, similar(tmp, 0), similar(tmp, 0)
         , parent, weights, tmp)
end

function Base.copy(S::NormalOpImpl)
  return NormalOpImpl(copy(S.parent), S.weights, copy(S.tmp))
end

"""
  normalOperator(parent (, weights); kwargs...)

  Constructs a normal operator of the parent in an opinionated way, i.e. it tries to apply optimisations to the resulting operator.
"""
function normalOperator(parent, weights=nothing; kwargs...)
  return NormalOp(eltype(storage_type((parent))); parent = parent, weights = weights)
end