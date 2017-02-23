import Colors
import Images
import MNIST
import NMF
import Origami
import ThreeQ
import PyPlot

function showimg(x)
	display(Colors.Gray.(reshape(x, 28, 28))); println()
end
function showimgs(x...)
	#=
	maxx = maximum(map(y->maximum(y), x))
	x = map(y->y / maxx, x)
	display(Colors.Gray.(hcat(map(y->reshape(y, 28, 28), x)...))); println()
	=#
	fig, axs = PyPlot.subplots(length(x))
	for i = 1:length(x)
		axs[i][:plot](x[i])
	end
	display(fig); println()
	PyPlot.close(fig)
end

halfnumsourcesignals = 5
numsourcesignals = 2 * halfnumsourcesignals
#sourcesignals = Array{Float64, 1}[]
timepoints = 200
sourcesignals = zeros(timepoints, numsourcesignals)
xs = linspace(0, 2 * pi, timepoints)
for i = 1:halfnumsourcesignals
	sourcesignals[:, 2 * i - 1] = sin.(2 ^ i * xs) + 1.0
	sourcesignals[:, 2 * i] =  cos.(2 ^ i * xs) + 1.0
end

srand(0)
numsmall = 100
mixingmatrix = rand([0, 1], numsourcesignals, numsmall)
imgs = sourcesignals * mixingmatrix
smallimgs = imgs[:, 1:numsmall]
#=
showimgs(map(i->sourcesignals[:, i], 1:size(sourcesignals, 2))...)
showimgs(map(i->smallimgs[:, i], 1:20)...)
=#

#=
for i = 1:timepoints
	estBrowi = Origami.solvesmalllsq(smallimgs, mixingmatrix, i; print_level=1)
	@show vecnorm(estBrowi - sourcesignals[i, :])
end
=#

numfeatures = numsourcesignals
@time nmfresult = NMF.nnmf(smallimgs, numfeatures)
nmfimgs = nmfresult.W * nmfresult.H
showimgs(map(i->nmfresult.W[:, i], 1:size(nmfresult.W, 2))...)
@show vecnorm(nmfimgs - smallimgs)
@show vecnorm(smallimgs)
nmfB = copy(nmfresult.W)
solver = ThreeQ.DWQMI.getdw2xsys4(mytoken)
#solver = ThreeQ.DWQMI.defaultsolver
num_reads = 1000
param_chain_factor = 1e2
estC = Origami.solvequbo(smallimgs, sourcesignals, solver; B=nmfB, qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, param_chain_factor=param_chain_factor)
#estC = Origami.solvequbo(smallimgs, sourcesignals, Origami.Qbsolv(); trueC=mixingmatrix, B=nmfB, qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, param_chain_factor=param_chain_factor)
numwrong = 0
for i = 1:size(estC, 2)
	if estC[:, i] != mixingmatrix[:, i]
		numwrong += 1
	end
end
@show numwrong / size(estC, 2)


#@time B, C = Origami.factor(smallimgs, numfeatures; trueC=mixingmatrix, B=sourcesignals, qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, param_chain_factor=param_chain_factor)
@time B, C = Origami.factor(smallimgs, numfeatures; qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, param_chain_factor=1e0)
showimgs(map(i->B[:, i], 1:size(B, 2))...)
A = B * C
for i = 1:5
	showimgs(smallimgs[:, i], A[:, i])
end
