# Load a pre-processed signal
using CSV
using DataFrames

df = CSV.read("data/b1382_clipped_ROI.txt", DataFrame)
ts = df.b1382_clipped_ROI


# Create windows
win_length = 400
win_offset = 200
num_wins = (length(ts) - win_length) ÷ win_offset
wins = Array{Int,2}(undef,num_wins,2)
for i in 1:num_wins
    wins[i,:] = [(i-1)*win_offset+1, (i-1)*win_offset + win_length]
end

# Let's plot something
# using CairoMakie
# CairoMakie.activate!()



# The loop
# using Statistics
using DelayEmbeddings
using Polynomials
using ComplexityMeasures
Tmax = 30 # maximum possible delay
results = Array{Any,2}(undef,num_wins,5)
for i = 1:num_wins
    println("Loop")
    X = ts[wins[i,1]:wins[i,2]]
    # X = ts[wins[1,1]:wins[1,2]]

    p = fit(1:length(X), X, 1).(1:length(X)) # normalising x
    X = X .- p
    X = (X .- minimum(X)) / (maximum(X) - minimum(X))

    theiler = estimate_delay(X, "mi_min") # estimate a Theiler window
    Y, τ_vals, ts_vals, Ls, εs = pecuzal_embedding(X; τs = 0:Tmax , w = theiler, econ = true) # estimate embedding
    m = length(τ_vals)
    shannon = entropy_permutation(Y, m=m)
    valbinsentr = information(Shannon(base=2), ValueBinning(0.01), X)

    results[i,:] = [Y, m, shannon, valbinsentr, τ_vals]
end


using CairoMakie
CairoMakie.activate!()

fig1 = Figure()
ax1 = Axis(fig1[1,1], title="embedding dimension m")
lines!(ax1, wins[:,1], results[:,2])

fig2 = Figure()
ax2 = Axis(fig2[1,1], title="Shannon entropy")
lines!(ax2, wins[:,1], results[:,3])

fig3 = Figure()
ax3 = Axis(fig3[1,1], title="Value binning Shannon entropy")
lines!(ax3, wins[:,1], results[:,4])