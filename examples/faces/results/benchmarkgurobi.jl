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
	timesfilename = joinpath(dir, "times_gurobi.jld")
	if !isfile(timesfilename)
		times = zeros(size(A, 2), length(Bs) - 1)
		numsolved = 0
		for k = 2:length(Bs)#start at 2 because the qubo solver never sees Bs[1]
			for i = 1:size(A, 2)
				Q = Origami.setupsmallqubo(A, Bs[k], i)
				target = ThreeQ.evalqubo(Q, Cs[k][:, i])
				for j = 1:100
					target += eps(Float32(target))#don't let floating point rounding error be the issue that prevents qbsolv from reaching the target
				end
				d = diag(Q)
				for j = 1:size(Q, 1)
					Q[j, j] = 0.0
				end
				m = ThreeQ.bqp(Q, d, Gurobi.GurobiSolver(BestObjStop=target, OutputFlag=0, Threads=1))
				times[i, k - 1] = JuMP.getsolvetime(m)
				numsolved += 1
				if i % 100 == 0
					@show sum(times)
					@show numsolved * numanneals * annealtime
					@show numsolved / length(times)
				end
				#=
				@show target
				@show JuMP.getobjectivevalue(m)
				@show target - JuMP.getobjectivevalue(m)
				@show JuMP.getsolvetime(m)
				=#
			end
		end
		@show sum(times)
		@show tqubos
		JLD.save(timesfilename, "times", times, "tlsqs", tlsqs, "tqubos", tqubos)
	end
end
