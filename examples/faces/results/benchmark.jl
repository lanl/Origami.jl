import Glob
import JLD
import Origami
import PyPlot
import ThreeQ

dirs = ["run1_40bits_crashed", "run2_40bits_finished", "run3_35bits_crashed", "run4_35bits_crashed", "run5_35bits_finished", "run6_35bits_10samples_crashed", "run7_35bits_crashed", "run8_35bits_crashed"]

A = JLD.load("faces.jld", "faces")
for dir in dirs
	filenames = Glob.glob("BnC_iteration_*.jld", dir)
	numanneals = parse(Int, split(split(filenames[1], "_")[end], ".")[1])
	@show numanneals
	annealtime = 20e-6
	Bs = Array(Array{Float64, 2}, length(filenames))
	Cs = Array(Array{Int, 2}, length(filenames))
	tlsqs = tqubos = zeros(0)
	for filename in filenames
		i = parse(Int, split(basename(filename), "_")[3]) + 1
		B, C, tlsqs, tqubos = JLD.load(filename, "B", "C", "tlsqs", "tqubos")
		Bs[i] = B
		Cs[i] = C
	end
	timesfilename = joinpath(dir, "times.jld")
	if !isfile(timesfilename)
		times = zeros(size(A, 2), length(Bs) - 1)
		for k = 2:length(Bs)#start at 2 because the qubo solver never sees Bs[1]
			for i = 1:size(A, 2)
				Q = Origami.setupsmallqubo(A, Bs[k], i)
				ThreeQ.writeqbsolvfile(Q, "test.qbsolvin")
				target = ThreeQ.evalqubo(Q, Cs[k][:, i])
				for j = 1:100
					target += eps(Float32(target))#don't let floating point rounding error be the issue that prevents qbsolv from reaching the target
				end
				timeout = 10 * 60
				thist = @elapsed bitsolution, output = ThreeQ.runqbsolv("laptop", "c4-sw_sample", "test.qbsolvin", size(Cs[k], 1), target, timeout; seed=0)
				times[i, k - 1] = parse(Float64, split(output[end])[1])
				if times[i, k - 1] > 1
					if ThreeQ.evalqubo(Q, bitsolution) > target
						times[i, k - 1] = timeout
					end
					@show thist - times[i, k - 1]
					@show ThreeQ.evalqubo(Q, bitsolution) > target
					@show i, k, thist, times[i, k - 1], sum(times)
					@show target, ThreeQ.evalqubo(Q, bitsolution), ThreeQ.evalqubo(Q, bitsolution) - target
					@show output[end]
					println(((k - 2) * size(A, 2) + i) / ((length(Bs) - 1) * size(A, 2)))
				end
			end
			@show sum(times)
		end
		JLD.save(timesfilename, "times", times, "tlsqs", tlsqs, "tqubos", tqubos)
	end
	numqubos = length(Cs) * size(Cs[1], 2)
	times, tlsqs, tqubos = JLD.load(timesfilename, "times", "tlsqs", "tqubos")
	fig, ax = PyPlot.subplots()
	ax[:plot](log10.(times[:]), ".", alpha=0.1)
	#ax[:plot](sort(log10.(times[:])))
	ax[:plot]([1, length(times)], log10.([numanneals*annealtime, numanneals*annealtime]))
	ax[:plot]([1, length(times)], log10.([tqubos[end] / numqubos, tqubos[end] / numqubos]))
	display(fig); println()
	PyPlot.close(fig)
	@show dir
	@show numanneals * annealtime * numqubos, tqubos[end], sum(times)
	@show sum(times .> numanneals * annealtime) / length(times)
	@show length(Cs)
	@show length(times)
end
