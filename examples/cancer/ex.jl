import Origami
import NMF

A = readcsv("cancer_mutation_data.csv")
if !isdefined(:solver)
	solver = ThreeQ.DWQMI.getdw2xsys4(mytoken)
	adjacency = ThreeQ.DWQMI.getadjacency(solver)
end

num_reads = 1000
num_repeats = 100
ks = eval(parse(ARGS[1]))
Bs = Dict()
Cs = Dict()
for k in ks
	if !isfile("results_$k.jld")
		Bs[k] = Array(Array{Float64, 2}, num_repeats)
		Cs[k] = Array(Array{Float64, 2}, num_repeats)
		for i = 1:num_repeats
			@show i, k
			B, C = Origami.factor(A, k; num_reads=num_reads, token=mytoken, adjacency=adjacency, qubosolver=solver, regularization=0, param_chain_factor=1e-2)
			Bs[k][i] = B
			Cs[k][i] = C
			writecsv("csvs/W_$(k)_$(lpad(i, 5, 0)).csv", B)
			writecsv("csvs/H_$(k)_$(lpad(i, 5, 0)).csv", C)
		end
		JLD.save("results_$k.jld", "Bs", Bs, "Cs", Cs)
	end
	Bs[k], Cs[k] = JLD.load("results_$k.jld", "Bs", "Cs")
end
for k in ks
end
