import Glob
import Gurobi
import JLD
import Origami
import PyPlot
import ThreeQ

#dirs = ["run1_40bits_crashed", "run2_40bits_finished", "run3_35bits_crashed", "run4_35bits_crashed", "run5_35bits_finished", "run6_35bits_10samples_crashed", "run7_35bits_crashed", "run8_35bits_crashed"]
dirs = ["run6_35bits_10samples_crashed", "run4_35bits_crashed", "run7_35bits_crashed", "run8_35bits_crashed"]
#dirs = ["run8_35bits_crashed"]

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
	for k = 2:length(Bs)#start at 2 because the qubo solver never sees Bs[1]
		for i = 1:250:size(A, 2)
			Q = Origami.setupsmallqubo(A, Bs[k], i)
			target = ThreeQ.evalqubo(Q, Cs[k][:, i])
			metadata = Dict()
			metadata["target"] = target
			@show target
			problemname = "$(dir[1:4])_$(k)_$i"
			metadata["problem_name"] = problemname
			ThreeQ.savebqpjson(Q, "carleton/$problemname.json")
		end
	end
end
