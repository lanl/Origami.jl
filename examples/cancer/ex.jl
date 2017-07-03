import Origami
import NMF

Aprenorm = readcsv("cancer_mutation_data.csv")
A = copy(Aprenorm)
for i = 1:size(A, 2)
	A[:, i] /= sum(A[:, i])
end

if !isdefined(:solver)
	solver = ThreeQ.DWQMI.getdw2xsys4(mytoken)
	adjacency = ThreeQ.DWQMI.getadjacency(solver)
end

num_reads = 1000
num_repeats = 100
if length(ARGS) > 0
	ks = eval(parse(ARGS[1]))
else
	ks = collect(3:7)
end
Bs = Dict()
Cs = Dict()
for k in ks
	if !isfile("results_$k.jld")
		Bs[k] = Array(Array{Float64, 2}, num_repeats)
		Cs[k] = Array(Array{Float64, 2}, num_repeats)
		for i = 1:num_repeats
			@show i, k
			B, C = Origami.factor(A, k; num_reads=num_reads, token=mytoken, adjacency=adjacency, qubosolver=solver, regularization=1e0, param_chain_factor=1e-2)
			Bs[k][i] = B
			Cs[k][i] = C
			writecsv("normcsvs/W_$(k)_$(lpad(i, 5, 0)).csv", B)
			writecsv("normcsvs/H_$(k)_$(lpad(i, 5, 0)).csv", C)
		end
		JLD.save("results_$k.jld", "Bs", Bs, "Cs", Cs)
	end
	t1, t2 = JLD.load("results_$k.jld", "Bs", "Cs")
	Bs[k] = t1[k]
	Cs[k] = t2[k]
end
for k in ks
	bestind = 0
	bestnorm = Inf
	for i = 1:length(Bs[k])
		thisnorm = vecnorm(A - Bs[k][i] * Cs[k][i])
		if thisnorm < bestnorm
			bestnorm = thisnorm
			bestind = i
		end
	end
	W = readcsv("normcsvs/W_$(k)_$(lpad(bestind, 5, 0)).csv")
	H = readcsv("normcsvs/H_$(k)_$(lpad(bestind, 5, 0)).csv")
	#@show bestnorm, "normcsvs/W_$(k)_$(lpad(bestind, 5, 0)).csv", "normcsvs/H_$(k)_$(lpad(bestind, 5, 0)).csv"
	println("$k components: norm=$(bestnorm), normcsvs/W_$(k)_$(lpad(bestind, 5, 0)).csv, normcsvs/H_$(k)_$(lpad(bestind, 5, 0)).csv")
end
