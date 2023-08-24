function testDCT1d(N=32)
  Random.seed!(1235)
  x = zeros(ComplexF64, N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2)) .+ 1im*rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  D1 = constructLinearOperator(DCTOp{ComplexF64}, shape=(N^2,), dcttype=2)
  D2 = sqrt(2/N^2)*[cos(pi/(N^2)*j*(k+0.5)) for j=0:N^2-1,k=0:N^2-1]
  D2[1,:] .*= 1/sqrt(2)
  D3 = constructLinearOperator(DCTOp{ComplexF64}, shape=(N^2,), dcttype=4)
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
end

function testFFT1d(N=32,shift=true)
  Random.seed!(1234)
  x = zeros(N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  D1 = constructLinearOperator(FFTOp{ComplexF64}, shape=(N^2,), shift=shift)
  D2 =  1.0/N*[exp(-2*pi*im*j*k/N^2) for j=0:N^2-1,k=0:N^2-1]

  y1 = D1*x
  if shift
    y2 = fftshift(D2*fftshift(x))
  else
    y2 = D2*x
  end
  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  x1 = adjoint(D1) * y1
  if shift
    x2 = ifftshift(adjoint(D2)*ifftshift(y2))
  else
    x2 = adjoint(D2)*y2
  end
  @test norm(x1 - x2) / norm(x1) ≈ 0 atol=0.01
end

function testFFT2d(N=32,shift=true)
  Random.seed!(1234)
  x = zeros(N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  D1 = constructLinearOperator(FFTOp{ComplexF64}, shape=(N,N), shift=shift)

  idx = CartesianIndices((N,N))[collect(1:N^2)]
  D2 =  1.0/N*[ exp(-2*pi*im*((idx[j][1]-1)*(idx[k][1]-1)+(idx[j][2]-1)*(idx[k][2]-1))/N) for j=1:N^2, k=1:N^2 ]

  y1 = D1*x
  if shift
    y2 = D2*vec(fftshift(reshape(x,N,N)))
    y2 = vec(fftshift(reshape(y2,N,N)))
  else
    y2 = D2*x
  end
  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  x1 = adjoint(D1) * y1
  if shift
    x2 = adjoint(D2)*vec(ifftshift(reshape(y2,N,N)))
    x2 = vec(ifftshift(reshape(x2,N,N)))
  else
    x2 = adjoint(D2)*y2
  end
  @test norm(x1 - x2) / norm(x1) ≈ 0 atol=0.01
end

function testWeighting(N=512)
  Random.seed!(1234)
  x1 = rand(N)
  weights = rand(N)
  W = WeightingOp(weights)
  y1 = W*x1
  y = weights .* x1

  @test norm(y1 - y) / norm(y) ≈ 0 atol=0.01

  x2 = rand(2*N)
  W2 = WeightingOp(weights,2)
  y2 = W2*x2
  y = repeat(weights,2) .* x2

  @test norm(y2 - y) / norm(y) ≈ 0 atol=0.01
end

function testGradOp1d(N=512)
  x = rand(N)
  G = GradientOp(eltype(x),size(x))
  G0 = Bidiagonal(ones(N),-ones(N-1), :U)[1:N-1,:]

  y = G*x
  y0 = G0*x
  @test norm(y - y0) / norm(y0) ≈ 0 atol=0.001

  xr = transpose(G)*y
  xr0 = transpose(G0)*y0

  @test norm(y - y0) / norm(y0) ≈ 0 atol=0.001
end

function testGradOp2d(N=64)
  x = repeat(1:N,1,N)
  G = GradientOp(eltype(x),size(x))
  G_1d = Bidiagonal(ones(N),-ones(N-1), :U)[1:N-1,:]

  y = G*vec(x)
  y0 = vcat( vec(G_1d*x), vec(x*transpose(G_1d)) )
  @test norm(y - y0) / norm(y0) ≈ 0 atol=0.001

  xr = transpose(G)*y
  y0_x = reshape(y0[1:N*(N-1)],N-1,N)
  y0_y = reshape(y0[N*(N-1)+1:end],N,N-1)
  xr0 = transpose(G_1d)*y0_x + y0_y*G_1d
  xr0 = vec(xr0)

  @test norm(xr - xr0) / norm(xr0) ≈ 0 atol=0.001
end

function testSampling(N=64)
  x = rand(ComplexF64,N,N)
  # index-based sampling
  idx = shuffle(collect(1:N^2)[1:N*div(N,2)])
  SOp = SamplingOp(idx,(N,N))
  y = SOp*vec(x)
  x2 = adjoint(SOp)*y
  # mask-based sampling
  msk = zeros(Bool,N*N);msk[idx].=true
  SOp2 = SamplingOp(msk)
  y2 = ComplexF64.(SOp2*vec(x))
  # references
  y_ref = vec(x[idx])
  x2_ref = zeros(ComplexF64,N^2)
  x2_ref[idx] .= y_ref
  # perform tests
  @test norm(y - y_ref) / norm(y_ref) ≈ 0 atol=0.000001
  @test norm(x2 - x2_ref) / norm(x2_ref) ≈ 0 atol=0.000001
  @test norm(y2 - x2_ref) / norm(x2_ref) ≈ 0 atol=0.000001
end

function testWavelet(M=64,N=60)
  x = rand(M,N)
  WOp = constructLinearOperator(WaveletOp{Float64}, shape=(M,N))
  x_wavelet = WOp*vec(x)
  x_reco = reshape( adjoint(WOp)*x_wavelet, M, N)

  @test norm(x_reco - x) / norm(x) ≈ 0 atol=0.001
end

@testset "Linear Operators" begin
  @info "test DCT-II and DCT-IV"
  for N in [2,8,16,32]
    testDCT1d(N)
  end
  @info "test FFT"
  for N in [8,16,32]
    testFFT1d(N,false)
    testFFT1d(N,true)
    testFFT2d(N,false)
    testFFT2d(N,true)
  end
  @info "test Weighting"
  testWeighting(512)
  @info "test gradientOp"
  testGradOp1d(512)
  testGradOp2d(64) 
  @info "test sampling"
  testSampling(64)
  @info "test WaveletOp"
  testWavelet(64,64)
  testWavelet(64,60)
end
