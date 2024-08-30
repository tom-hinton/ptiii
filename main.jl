using CSV
using DataFrames

df = CSV.read("data/b1382_clipped_ROI.txt", DataFrame)
s = df.b1382_clipped_ROI

using DelayEmbeddings
theiler = estimate_delay(s, "mi_min") # estimate a Theiler window
Tmax = 30 # maximum possible delay

Y, τ_vals, ts_vals, Ls, εs = pecuzal_embedding(s; τs = 0:Tmax , w = theiler, econ = true)

println("τ_vals = ", τ_vals)
println("Ls = ", Ls)
println("L_total_uni: $(sum(Ls))")

using CairoMakie
fig = Figure()
axs = Axis3(fig[1,1])
lines!(axs, Y[:, 1], Y[:, 2], Y[:, 3])
fig