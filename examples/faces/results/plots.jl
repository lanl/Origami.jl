import Glob
import JLD
using LaTeXStrings
import Origami
import PyPlot
import ThreeQ

dirs = ["run6_35bits_10samples_crashed", "run4_35bits_crashed", "run7_35bits_crashed", "run8_35bits_crashed"]
times = Array(Array{Float64, 2}, length(dirs))
gtimes = Array(Array{Float64, 2}, length(dirs))
tlsqs = Array(Float64, length(dirs))
tqubos = Array(Float64, length(dirs))
Bss = Array(Array{Array{Float64, 2}}, length(dirs))
Css = Array(Array{Array{Int, 2}}, length(dirs))
numanneals = Array(Int, length(dirs))
numqubos = Array(Int, length(dirs))
annealtime = 20e-6

A = JLD.load("faces.jld", "faces")
for (j, dir) in enumerate(dirs)
	filenames = Glob.glob("BnC_iteration_*.jld", dir)
	numanneals[j] = parse(Int, split(split(filenames[1], "_")[end], ".")[1])
	Bss[j] = Array(Array{Float64, 2}, length(filenames))
	Css[j] = Array(Array{Int, 2}, length(filenames))
	for filename in filenames
		i = parse(Int, split(basename(filename), "_")[3]) + 1
		B, C = JLD.load(filename, "B", "C")
		Bss[j][i] = B
		Css[j][i] = C
	end
	numqubos[j] = length(Css[j]) * size(Css[j][1], 2)
	timesfilename = joinpath(dir, "times.jld")
	times[j], b, c = JLD.load(timesfilename, "times", "tlsqs", "tqubos")
	tlsqs[j] = b[end]
	tqubos[j] = c[end]
	gtimesfilename = joinpath(dir, "times_gurobi.jld")
	gtimes[j] = JLD.load(gtimesfilename, "times")
end
fig, axs = PyPlot.subplots(1, length(dirs), sharey=true, figsize=(8, 3), dpi=225)
axs[1][:set_ylabel](L"\log_{10}(TTT\ [s])")
l1, l2, l3 = 0, 0, 0
for i = 1:length(dirs)
	axs[i][:plot](log10.(times[i][:]), ".", alpha=0.05, color="#ce7058", label="qbsolv")
	axs[i][:plot](log10.(gtimes[i][:]), ".", alpha=0.05, color="#5a9bd4", label="Gurobi")
	axs[i][:plot]([1, length(times[i])], log10.([numanneals[i] * annealtime, numanneals[i] * annealtime]), alpha=1, linewidth=2, color="orange", label="D-Wave 2X")
	if i == 4
		axs[i][:xaxis][:set_ticks]([0, 5000, 15000])
	end
	if i == length(dirs)
		handles, labels = axs[i][:get_legend_handles_labels]()
		#display = [0, 1, 2]
		#axs[i][:legend](handles, labels)
		axs[i][:legend]([PyPlot.plt[:Line2D]((0, 1), (0, 0), color="#ce7058", marker=".", linestyle=""), PyPlot.plt[:Line2D]((0, 1), (0, 0), color="#5a9bd4", marker=".", linestyle=""), handles[3]], labels, bbox_to_anchor=(-1.2, -0.25), fancybox=true, ncol=3, loc="upper center")
		@show handles
		@show labels
		#axs[i][:legend](["qbsolv", "Gurobi", "D-Wave 2X"], bbox_to_anchor=(-1.2, -0.25), fancybox=true, ncol=3, loc="upper center")
	end
	axs[i][:set_ylim](-5, 3)
	axs[i][:set_xlim](1, length(times[i]))
	axs[i][:set_title]("$(numanneals[i]) anneals\n@20μs/anneal")
	axs[i][:set_xlabel]("QUBO number")
end
#fig[:legend]([l1, l2, l3], ["qbsolv", "Gurobi", "D-Wave 2X"], loc=(0.5, 0), ncol=3)
#fig[:legend]((l1, l2, l3), ("qbsolv", "Gurobi", "D-Wave 2X"), loc="lower center", ncol=3)
#fig[:legend]([l1, l2, l3], ["blah", "blee", "args"])
fig[:tight_layout]()
fig[:savefig]("performance.png")
display(fig); println()
PyPlot.close(fig)
println("qbsolv:")
println("#ann\t%t_anneal < t_qbsolv\tannealing speedup\ttotal speedup w/IO\t#maxt\tann_t\tann_t (total)\tqbsolv_t (total)")
for i = 1:length(dirs)
	println("$(numanneals[i])\t$(sum(times[i] .> (numanneals[i] * annealtime)) / length(times[i]))\t$(sum(times[i]) / (numanneals[i] * annealtime * numqubos[i]))\t$(sum(times[i]) / tqubos[i])\t$(sum(times[i] .> 599))\t$(numanneals[i] * annealtime * numqubos[i])\t$(tqubos[i])\t$(sum(times[i]))")
end
println("Gurobi:")
println("#ann\t%t_anneal < t_qbsolv\tannealing speedup\ttotal speedup w/IO\t#maxt\tann_t\tann_t (total)\tqbsolv_t (total)")
for i = 1:length(dirs)
	println("$(numanneals[i])\t$(sum(gtimes[i] .> (numanneals[i] * annealtime)) / length(gtimes[i]))\t$(sum(gtimes[i]) / (numanneals[i] * annealtime * numqubos[i]))\t$(sum(gtimes[i]) / tqubos[i])\t$(sum(gtimes[i] .> 599))\t$(numanneals[i] * annealtime * numqubos[i])\t$(tqubos[i])\t$(sum(gtimes[i]))")
end
fig2, ax2 = PyPlot.subplots(1, 1)
ax2[:plot](log10.(numanneals), map(x->log10(sum(x)), times), ".-", color="#ce7058", label="qbsolv", ms=15)
ax2[:plot](log10.(numanneals), map(x->log10(sum(x)), gtimes), ".-", color="#5a9bd4", label="Gurobi", ms=15)
ax2[:plot](log10.(numanneals), log10.(numanneals .* map(length, times) * annealtime), ".-", color="orange", label="annealing time", ms=15)
ax2[:set_xlabel](L"\log_{10}(" * "number of anneals" * L")")
ax2[:set_ylabel](L"\log_{10}(" * "cumulative TTT [s]" * L")")
ax2[:legend]()
fig2[:tight_layout]()
fig2[:savefig]("cumulativeperformance.pdf")
display(fig2); println(); PyPlot.close(fig2)
