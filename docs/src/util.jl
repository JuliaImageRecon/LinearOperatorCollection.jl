using CairoMakie, LinearOperatorCollection
function plot_image(figPos, img; title = "", width = 150, height = 150, colorscale = identity, colormap = :viridis)
  ax = CairoMakie.Axis(figPos[1, 1]; yreversed=true, title, width, height)
  hidedecorations!(ax)
  hm = heatmap!(ax, img, colorscale = colorscale, colormap = colormap)
  Colorbar(figPos[1, 2], hm)
end
N = 256
using ImagePhantoms, ImageGeoms
image = shepp_logan(N, SheppLoganToft())
nothing