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
	timesfilename = joinpath(dir, "times_scip.jld")
	if !isfile(timesfilename)
		times = fill(NaN, size(A, 2), length(Bs) - 1)
		for k = 2:length(Bs)#start at 2 because the qubo solver never sees Bs[1]
			for i = 1:size(A, 2)
				Q = Origami.setupsmallqubo(A, Bs[k], i)
				d = diag(Q)
				for j = 1:size(Q, 1)
					Q[j, j] = 0.0
				end
				m = ThreeQ.bqp(Q, d)
				times[i, k - 1] = JuMP.getsolvetime(m)
			end
		end
		JLD.save(timesfilename, "times", times, "tlsqs", tlsqs, "tqubos", tqubos)
	end
end
