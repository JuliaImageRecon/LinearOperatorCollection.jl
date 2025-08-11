using Documenter, Literate # Documentation
using RadonKA, Wavelets, NFFT, FFTW # Extensions
using CairoMakie, ImageGeoms, ImagePhantoms # Documentation Example Packages

# Generate examples
OUTPUT_BASE = joinpath(@__DIR__(), "src", "generated")
INPUT_BASE = joinpath(@__DIR__(), "src", "literate")
for (root, dirs, files) in walkdir(INPUT_BASE)
    for dir in dirs
        OUTPUT = joinpath(OUTPUT_BASE, dir)
        INPUT = joinpath(INPUT_BASE, dir)
        for file in filter(f -> endswith(f, ".jl"), readdir(INPUT))
            Literate.markdown(joinpath(INPUT, file), OUTPUT)
        end
    end
end

makedocs(
    format = Documenter.HTML(prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/JuliaImageRecon/LinearOperatorCollection.jl",
        assets=String[],
        collapselevel=1,
    ),
    modules = [LinearOperatorCollection],
    sitename = "LinearOperatorCollection",
    authors = "Tobias Knopp, Niklas Hackelberg and Contributors",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "generated/tutorials/overview.md",
        "Tutorials" => Any[
            "Weighting Operator" => "generated/tutorials/weighting.md",
            "Product Operator" => "generated/tutorials/product.md",
            "Diagonal Operator" => "generated/tutorials/diagonal.md",
            "Normal Operator" => "generated/tutorials/normal.md",
            "Gradient Operator" => "generated/tutorials/gradient.md",
            "FFT Operator" => "generated/tutorials/fft.md",
            "NFFT Operator" => "generated/tutorials/nfft.md",
            "Wavelet Operator" => "generated/tutorials/wavelet.md",
            "Radon Operator" => "generated/tutorials/radon.md"
        ],
        "How to" => Any[
            "Implement Custom Operators" => "generated/howtos/custom.md",
            "Enable GPU Acceleration" => "generated/howtos/gpu.md",
        ],
        "Explanations" => Any[
            "Operator Structure" => "operators.md",
        ],
        "Reference" => "references.md"
    ],
    warnonly = [:missing_docs]
)

deploydocs(repo   = "github.com/MagneticParticleImaging/MPIReco.jl.git",
           target = "build")
