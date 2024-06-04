function testDCT1d(N=32;arrayType = Array)
  Random.seed!(1235)
  x = zeros(ComplexF64, N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2)) .+ 1im*rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  D1 = DCTOp(ComplexF64, shape=(N^2,), dcttype=2)
  D2 = sqrt(2/N^2)*[cos(pi/(N^2)*j*(k+0.5)) for j=0:N^2-1,k=0:N^2-1]
  D2[1,:] .*= 1/sqrt(2)
  D3 = DCTOp(ComplexF64, shape=(N^2,), dcttype=4)
  D4 = sqrt(2/N^2)*[cos(pi/(N^2)*(j+0.5)*(k+0.5)) for j=0:N^2-1,k=0:N^2-1]

  y1 = D1*x
  y2 = D2*x

  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  y3 = D3*x
  y4 = D4*x
  @test norm(y3 - y4) / norm(y1) ≈ 0 atol=0.01

  x1 = adjoint(D1)*y1
  @test norm(x1 - x) / norm(x) ≈ 0 atol=0.01

  x2 = adjoint(D3)*y3
  @test norm(x2 - x) / norm(x) ≈ 0 atol=0.01
  true
end

function testFFT1d(N=32,shift=true;arrayType = Array)
  Random.seed!(1234)
  x = zeros(N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  xop = arrayType(x)
  D1 = FFTOp(ComplexF64, shape=(N^2,), shift=shift, S = typeof(ComplexF64.(xop)))
  D2 =  1.0/N*[exp(-2*pi*im*j*k/N^2) for j=0:N^2-1,k=0:N^2-1]

  y1 = Array(D1*xop)
  if shift
    y2 = fftshift(D2*fftshift(x))
  else
    y2 = D2*x
  end
  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  x1 = Array(adjoint(D1) * arrayType(y1))
  if shift
    x2 = ifftshift(adjoint(D2)*ifftshift(y2))
  else
    x2 = adjoint(D2)*y2
  end
  @test norm(x1 - x2) / norm(x1) ≈ 0 atol=0.01
  true
end

function testFFT2d(N=32,shift=true;arrayType = Array)
  Random.seed!(1234)
  x = zeros(N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  xop = arrayType(x)
  D1 = FFTOp(ComplexF64, shape=(N,N), shift=shift, S = typeof(ComplexF64.(xop)))

  idx = CartesianIndices((N,N))[collect(1:N^2)]
  D2 =  1.0/N*[ exp(-2*pi*im*((idx[j][1]-1)*(idx[k][1]-1)+(idx[j][2]-1)*(idx[k][2]-1))/N) for j=1:N^2, k=1:N^2 ]

  y1 = Array(D1*xop)
  if shift
    y2 = D2*vec(fftshift(reshape(x,N,N)))
    y2 = vec(fftshift(reshape(y2,N,N)))
  else
    y2 = D2*x
  end
  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  x1 = Array(adjoint(D1) * arrayType(y1))
  if shift
    x2 = adjoint(D2)*vec(ifftshift(reshape(y2,N,N)))
    x2 = vec(ifftshift(reshape(x2,N,N)))
  else
    x2 = adjoint(D2)*y2
  end
  @test norm(x1 - x2) / norm(x1) ≈ 0 atol=0.01
  true
end

function testWeighting(N=512;arrayType = Array)
  Random.seed!(1234)
  x1 = rand(N)
  weights = rand(N)
  W = WeightingOp(arrayType(weights))
  y1 = W*arrayType(x1)
  y = weights .* x1

  @test norm(Array(y1) - y) / norm(y) ≈ 0 atol=0.01

  x2 = rand(2*N)
  W2 = WeightingOp(arrayType(weights), 2)
  y2 = W2*arrayType(x2)
  y = repeat(weights,2) .* x2

  @test norm(Array(y2) - y) / norm(y) ≈ 0 atol=0.01
  true
end

function testGradOp1d(N=512;arrayType = Array)
  x = rand(N)
  xop = arrayType(x)
  G = GradientOp(eltype(x); shape=size(x), S = typeof(xop))
  G0 = Bidiagonal(ones(N),-ones(N-1), :U)[1:N-1,:]

  y = Array(G*xop)
  y0 = G0*x
  @test norm(y - y0) / norm(y0) ≈ 0 atol=0.001

  xr = Array(transpose(G)*arrayType(y))
  xr0 = transpose(G0)*y0

  @test norm(xr - xr0) / norm(xr0) ≈ 0 atol=0.001
  true
end

function testGradOp2d(N=64;arrayType = Array)
  x = repeat(1:N,1,N)
  xop = arrayType(vec(x))
  G = GradientOp(eltype(x); shape=size(x), S = typeof(xop))
  G_1d = Bidiagonal(ones(N),-ones(N-1), :U)[1:N-1,:]

  y = Array(G*xop)
  y0 = vcat( vec(G_1d*x), vec(x*transpose(G_1d)) )
  @test norm(y - y0) / norm(y0) ≈ 0 atol=0.001

  xr = Array(transpose(G)*arrayType(y))
  y0_x = reshape(y0[1:N*(N-1)],N-1,N)
  y0_y = reshape(y0[N*(N-1)+1:end],N,N-1)
  xr0 = transpose(G_1d)*y0_x + y0_y*G_1d
  xr0 = vec(xr0)

  @test norm(xr - xr0) / norm(xr0) ≈ 0 atol=0.001
  true
end

function testDirectionalGradOp(N=64;arrayType = Array)
  x = rand(ComplexF64,N,N)
  xop = arrayType(vec(x))
  G1 = GradientOp(eltype(x); shape=size(x), dims=1, S = typeof(xop))
  G2 = GradientOp(eltype(x); shape=size(x), dims=2, S = typeof(xop))
  G_1d = Bidiagonal(ones(N),-ones(N-1), :U)[1:N-1,:]

  y1 = Array(G1*xop)
  y2 = Array(G2*xop)
  y1_ref = zeros(ComplexF64, N-1,N)
  y2_ref = zeros(ComplexF64, N, N-1)
  for i=1:N
    y1_ref[:,i] .= G_1d*x[:,i]
    y2_ref[i,:] .= G_1d*x[i,:]
  end

  @test norm(y1-vec(y1_ref)) / norm(y1_ref) ≈ 0 atol=0.001
  @test norm(y2-vec(y2_ref)) / norm(y2_ref) ≈ 0 atol=0.001
  
  x1r = Array(transpose(G1)*arrayType(y1))
  x2r = Array(transpose(G2)*arrayType(y2))

  x1r_ref = zeros(ComplexF64, N,N)
  x2r_ref = zeros(ComplexF64, N,N)
  for i=1:N
    x1r_ref[:,i] .= transpose(G_1d)*y1_ref[:,i]
    x2r_ref[i,:] .= transpose(G_1d)*y2_ref[i,:]
  end
  @test norm(x1r-vec(x1r_ref)) / norm(x1r_ref) ≈ 0 atol=0.001
  @test norm(x2r-vec(x2r_ref)) / norm(x2r_ref) ≈ 0 atol=0.001
  true
end

function testSampling(N=64;arrayType = Array)
  x = rand(ComplexF64,N,N)
  xop = arrayType(vec(x))
  # index-based sampling
  idx = shuffle(collect(1:N^2)[1:N*div(N,2)])
  SOp = SamplingOp(ComplexF64, pattern=idx, shape=(N,N), S = typeof(xop))
  y = Array(SOp*xop)
  x2 = Array(adjoint(SOp)*arrayType(y))
  # mask-based sampling
  msk = zeros(Bool,N*N);msk[idx].=true
  SOp2 = SamplingOp(ComplexF64, pattern=msk, S = typeof(xop))
  y2 = ComplexF64.(Array(SOp2*xop))
  # references
  y_ref = vec(x[idx])
  x2_ref = zeros(ComplexF64,N^2)
  x2_ref[idx] .= y_ref
  # perform tests
  @test norm(y - y_ref) / norm(y_ref) ≈ 0 atol=0.000001
  @test norm(x2 - x2_ref) / norm(x2_ref) ≈ 0 atol=0.000001
  @test norm(y2 - x2_ref) / norm(x2_ref) ≈ 0 atol=0.000001
  true
end

function testWavelet(M=64,N=60;arrayType = Array)
  x = rand(M,N)
  xop = arrayType(vec(x))
  WOp = WaveletOp(Float64, shape=(M,N), S = typeof(xop))
  # TODO comparison against wavelet?
  x_wavelet = Array(WOp*xop)
  x_reco = reshape( adjoint(WOp)*x_wavelet, M, N)

  @test norm(x_reco - x) / norm(x) ≈ 0 atol=0.001
  true
end

# test FourierOperators
function testNFFT2d(N=16;arrayType = Array)
  # random image
  x = zeros(ComplexF64,N,N)
  for i=1:N,j=1:N
    x[i,j] = rand()
  end

  # FourierMatrix
  idx = CartesianIndices((N,N))[collect(1:N^2)]
  F = [ exp(-2*pi*im*((idx[j][1]-1)*(idx[k][1]-1)+(idx[j][2]-1)*(idx[k][2]-1))/N) for j=1:N^2, k=1:N^2 ]
  F_adj = adjoint(F)

  # Operator
  xop = arrayType(vec(x))
  nodes = [(idx[d] - N÷2 - 1)./N for d=1:2, idx in vec(CartesianIndices((N,N)))]
  F_nfft = NFFTOp(ComplexF64; shape=(N,N), nodes, symmetrize=false, S = typeof(xop))

  # test against FourierOperators
  y = vec( ifftshift(reshape(F*vec(fftshift(x)),N,N)) )
  y_adj = vec( ifftshift(reshape(F_adj*vec(fftshift(x)),N,N)) )

  y_nfft = Array(F_nfft * xop)
  y_adj_nfft = Array(adjoint(F_nfft) * xop)

  @test y     ≈ y_nfft      rtol = 1e-2
  @test y_adj ≈ y_adj_nfft  rtol = 1e-2

  # test AHA w/o Toeplitz
  F_nfft.toeplitz = false
  AHA = normalOperator(F_nfft)
  y_AHA_nfft = Array(AHA * xop)
  y_AHA = F' * F * xop
  @test y_AHA ≈ y_AHA_nfft   rtol = 1e-2

  # test AHA with Toeplitz
  F_nfft.toeplitz = true
  AHA = normalOperator(F_nfft)
  y_AHA_nfft_1 = Array(AHA * xop)
  y_AHA_nfft_2 = Array(adjoint(F_nfft) * F_nfft * xop)
  y_AHA = F' * F * vec(x)
  @test y_AHA_nfft_1 ≈ y_AHA_nfft_2   rtol = 1e-2
  @test y_AHA ≈ y_AHA_nfft_1   rtol = 1e-2

  # test type stability;
  # TODO: Ensure type stability for Trajectory objects and test here
  nodes = Float32.(nodes)
  F_nfft = NFFTOp(ComplexF32; shape=(N,N), nodes, symmetrize=false, S = typeof(ComplexF32.(xop)))

  y_nfft = F_nfft * ComplexF32.(xop)
  y_adj_nfft = adjoint(F_nfft) * ComplexF32.(xop)

  @test Complex{eltype(nodes)} === eltype(y_nfft)
  @test Complex{eltype(nodes)} === eltype(y_adj_nfft)
  true
end

function testNFFT3d(N=12;arrayType = Array)
  # random image
  x = zeros(ComplexF64,N,N,N)
  for i=1:N,j=1:N,k=1:N
    x[i,j,k] = rand()
  end

  # FourierMatrix
  idx = CartesianIndices((N,N,N))[collect(1:N^3)]
  F = [ exp(-2*pi*im*((idx[j][1]-1)*(idx[k][1]-1)+(idx[j][2]-1)*(idx[k][2]-1)+(idx[j][3]-1)*(idx[k][3]-1))/N) for j=1:N^3, k=1:N^3 ]
  F_adj = F'

  # Operator
  xop = arrayType(vec(x))
  nodes = [(idx[d] - N÷2 - 1)./N for d=1:3, idx in vec(CartesianIndices((N,N,N)))]
  F_nfft = NFFTOp(ComplexF64; shape=(N,N,N), nodes=nodes, symmetrize=false, S = typeof(xop))

  # test agains FourierOperators
  y = vec( ifftshift(reshape(F*vec(fftshift(x)),N,N,N)) )
  y_adj = vec( ifftshift(reshape(F_adj*vec(fftshift(x)),N,N,N)) )

  y_nfft = Array(F_nfft*xop)
  y_adj_nfft = Array(adjoint(F_nfft) * xop)

  @test  y     ≈ y_nfft     rtol = 1e-2
  @test  y_adj ≈ y_adj_nfft rtol = 1e-2

  # test AHA w/o Toeplitz
  F_nfft.toeplitz = false
  AHA = normalOperator(F_nfft)
  y_AHA_nfft = Array(AHA * xop)
  y_AHA = F' * F * vec(x)
  @test y_AHA ≈ y_AHA_nfft   rtol = 1e-2

  # test AHA with Toeplitz
  F_nfft.toeplitz = true
  AHA = normalOperator(F_nfft)
  y_AHA_nfft_1 = Array(AHA * xop)
  y_AHA_nfft_2 = Array(adjoint(F_nfft) * F_nfft * xop)
  y_AHA = F' * F * vec(x)
  @test y_AHA_nfft_1 ≈ y_AHA_nfft_2   rtol = 1e-2
  @test y_AHA ≈ y_AHA_nfft_1   rtol = 1e-2
  true
end

# TODO RadonOp

@testset "Linear Operators" begin
  @testset for arrayType in arrayTypes
    @info "test DCT-II and DCT-IV Ops: $arrayType"
    for N in [2,8,16,32]
      @test testDCT1d(N;arrayType) skip = arrayType != Array # Not implemented for GPUs
    end
    @info "test FFTOp: $arrayType"
    for N in [8,16,32]
      @test testFFT1d(N,false;arrayType)
      @test testFFT1d(N,true;arrayType)
      @test testFFT2d(N,false;arrayType)
      @test testFFT2d(N,true;arrayType)
    end
    @info "test WeightingOp: $arrayType"
    @test testWeighting(512;arrayType)
    @info "test GradientOp: $arrayType"
    @test testGradOp1d(512;arrayType)
    @test testGradOp2d(64;arrayType)
    @test testDirectionalGradOp(64;arrayType) 
    @info "test SamplingOp: $arrayType"
    @test testSampling(64;arrayType)
    @info "test WaveletOp: $arrayType"
    @test testWavelet(64,64;arrayType)
    @test testWavelet(64,60;arrayType)
    @info "test NFFTOp: $arrayType"
    @test testNFFT2d(;arrayType) skip = arrayType == JLArray # JLArray does not have a NFFTPlan
    @test testNFFT3d(;arrayType) skip = arrayType == JLArray # JLArray does not have a NFFTPlan
  end
end
