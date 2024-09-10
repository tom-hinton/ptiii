run_code = "10"


# LOAD OPTS
using JSON
results_dir = pwd() * "/results/" * run_code
optspath = results_dir * "/opts.json"
opts = JSON.parsefile(optspath)


# LOAD (PRE-PROCESSED) SIGNAL
using CSV
using DataFrames

println("Reading CSV...")
csvtspath = pwd() * "/data/" * opts["exp_code"] * "_" * string(opts["sampling_freq"]) * "hz_ts.csv"
data = CSV.read(csvtspath, DataFrame)
# data = CSV.read("data/b1383_20hz_ts.csv", DataFrame)
ts = data."SHEAR STRESS"
println("... read.")


# GENERATE OR LOAD WINDOWS
wins = opts["windows"]
# win_length = 3000
# win_offset = 1500
# num_wins = (length(ts) - win_length) ÷ win_offset
# wins = Array{Int,2}(undef,num_wins,2)
# for i in 1:num_wins
#     wins[i,:] = [(i-1)*win_offset+1, (i-1)*win_offset + win_length]
# end


# THE LOOP
using DelayEmbeddings
using Polynomials
using ComplexityMeasures
using JSONTables
Tmax = opts["dynsys_params"]["tau_max_kraemer"]
valbins = opts["dynsys_params"]["valbins"]
# Tmax = 40 # maximum possible delay
# valbins=0.005
df = DataFrame(m_kraemer=Int64[], tau_vals=Any[], Ls=Any[], epsilons=Any[], shannon_entropy_vb=Float64[])
for i in axes(wins, 1)
    println("Loop "*string(i)*"/"*string(size(wins,1)))
    X = ts[(wins[i][1]+1):(wins[i][2]+1)]
    # X = ts[wins[1,1]:wins[1,2]]

    p = fit(1:length(X), X, 1).(1:length(X)) # normalising x
    X = X .- p
    X = (X .- minimum(X)) / (maximum(X) - minimum(X))

    theiler = estimate_delay(X, "mi_min") # estimate a Theiler window
    Y, tau_vals, ts_vals, Ls, epsilons = pecuzal_embedding(X; τs = 0:Tmax , w = theiler, econ = true) # calculate embedding
    m = length(tau_vals)

    shannon_entropy = information(Shannon(base=2), ValueBinning(valbins), X)

    push!(df, (m, tau_vals, Ls, epsilons, shannon_entropy))

    # SAVE OUTPUT TO JSON
    output_juliapath = results_dir * "/output_julia.json"
    # output_juliapath = pwd() * "/results/" * run_code * "/output_julia.json"
    open(output_juliapath, "w") do file
        write(file, objecttable(df))
    end

end


# using CairoMakie
# CairoMakie.activate!()

# fig1 = Figure()
# ax1 = Axis(fig1[1,1], title="embedding dimension m")
# lines!(ax1, wins[:,1], results[:,2])

# fig2 = Figure()
# ax2 = Axis(fig2[1,1], title="Shannon entropy")
# lines!(ax2, wins[:,1], results[:,3])

# fig3 = Figure()
# ax3 = Axis(fig3[1,1], title="Value binning Shannon entropy")
# lines!(ax3, wins[:,1], results[:,4])