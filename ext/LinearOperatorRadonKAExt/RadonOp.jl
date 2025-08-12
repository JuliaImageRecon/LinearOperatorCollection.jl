"""
    RadonOp(::Type{T}; shape::NTuple{N, Int}, angles, geometry = RadonParallelCircle(shape[1], -(shape[1]-1)÷2:(shape[1]-1)÷2), μ = nothing, S = Vector{T}) where {T, N}

Generates a `RadonOp` which evaluates the Radon transform operator and its adjoint (backprojection) for a given geometry and projection angles.

# Arguments:
* `T`                       - element type for the operator (e.g., `Float64`, `ComplexF32`)
* `shape::NTuple{N, Int}`   - size of the image
* `angles`                  - array of projection angles
* `geometry`                - Radon geometry descriptor (default: parallel beam circle)
* `μ`                       - optional attenuation map (for attenuated Radon transform)
* `S`                       - storage type for internal vectors (default: `Vector{T}`)
"""
function LinearOperatorCollection.RadonOp(::Type{T}; shape::NTuple{N, Int}, angles,
   geometry = RadonParallelCircle(shape[1], -(shape[1]-1)÷2:(shape[1]-1)÷2), μ = nothing, S = Vector{T}) where {T, N}
  return RadonOpImpl(T; shape, angles, geometry, μ, S)
end

mutable struct RadonOpImpl{T, vecT <: AbstractVector{T}, vecT2, G, A} <: RadonOp{T}
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
  angles :: vecT2
  geometry :: G
  μ :: A
end

LinearOperators.storage_type(op::RadonOpImpl) = typeof(op.Mv5)

function RadonOpImpl(T::Type; shape::NTuple{N, Int64}, angles, geometry, μ, S) where N
  N_sinogram = length(geometry.in_height)
  N_angles = length(angles)
  d = length(shape) == 3 ? shape[3] : 1
  nrow = N_sinogram * N_angles * d
  ncol = prod(shape)
  return RadonOpImpl(nrow, ncol, false, false,
  (res, x) -> prod_radon!(res, x, shape, angles, geometry, μ),
  nothing, 
  (res, x) -> ctprod_radon!(res, x, (N_sinogram, N_angles, d), angles, geometry, μ),
  0, 0, 0, true, false, true, S(undef, 0), S(undef, 0), angles, geometry, μ)
end

prod_radon!(res::vecT, x::vecT, shape, angles::vecT2, geometry::G, μ::A) where {vecT, vecT2, G, A} = copyto!(res, radon(reshape(x, shape), angles; geometry, μ))
prod_radon!(res::vecT, x::vecT, shape, angles::vecT2, ::Nothing, μ::A) where {vecT, vecT2, A} = copyto!(res, radon(reshape(x, shape), angles; μ))

ctprod_radon!(res::vecT, x::vecT, shape, angles::vecT2, geometry::G, μ::A) where {vecT, vecT2, G, A} = copyto!(res, RadonKA.backproject(reshape(x, shape), angles; geometry, μ))
ctprod_radon!(res::vecT, x::vecT, shape, angles::vecT2, ::Nothing, μ::A) where {vecT, vecT2, A} = copyto!(res, RadonKA.backproject(reshape(x, shape), angles; μ))
