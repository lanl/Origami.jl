import NMF
import Origami
import ThreeQ
import PyPlot

const tqubos = Float64[]
const tlsqs = Float64[]
function callback(B, C, i, tlsq, tqubo)
	push!(tlsqs, tlsq)
	push!(tqubos, tqubo)
	@show tlsqs
	@show tqubos
	JLD.save("BnC_iteration_$(i)_$(numsmall)_$(numfeatures)_$(num_reads).jld", "B", B, "C", C, "tqubos", tqubos, "tlsqs", tlsqs)
end

phases = collect(linspace(0, pi, 7))
modes = [1 / 4, .5, 1.0, 2.0, 4.0]
numsourcesignals = length(phases) * length(modes)
timepoints = 500
B = zeros(timepoints, numsourcesignals)
xs = linspace(0, 2 * pi, timepoints)
i = 1
for phase in phases, mode in modes
	B[:, i] = sin.(phase + mode * xs) + 1.0
	i += 1
end

srand(0)
numsmall = 10000
C = rand([0, 1], numsourcesignals, numsmall)
A = B * C

numfeatures = numsourcesignals
if !isdefined(:nmfA)
	println("nmf time:")
	@time nmfresult = NMF.nnmf(A, numfeatures)
	nmfB = nmfresult.W
	nmfC = nmfresult.H
	nmfA = nmfresult.W * nmfresult.H
	@show vecnorm(A - nmfA) / vecnorm(A)
end

if !isdefined(:solver)
	solver = ThreeQ.DWQMI.getdw2xsys4(mytoken)
	#solver = ThreeQ.DWQMI.defaultsolver
	adjacency = ThreeQ.DWQMI.getadjacency(solver)
end

num_reads = 100
@time dwB, dwC = Origami.factor(A, numfeatures; qubosolver=solver, num_reads=num_reads, min_iter=3, max_iter=5, embedding_dir=abspath("../faces/embeddings"), callback=callback, token=mytoken)
dwA = dwB * dwC
@show vecnorm(A - dwA) / vecnorm(A)
