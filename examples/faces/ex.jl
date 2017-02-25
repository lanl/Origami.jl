import Colors
import Images
import JLD
import NMF
import Origami
import ThreeQ

function rescaleB!(B)
	for j = 1:size(B, 2)
		m = maximum(B[:, j])
		for i = 1:size(B, 1)
			B[i, j] /= m
		end
	end
end
function showimg(x)
	display(Colors.Gray.(reshape(x, 19, 19))); println()
end
function showimgs(x...)
	display(Colors.Gray.(hcat(map(y->reshape(y, 19, 19), x)...))); println()
end

faces = JLD.load("faces.jld", "faces")
numsmall = 1000
smallfaces = faces[:, 1:numsmall]
numfeatures = 36
@time nmfresult = NMF.nnmf(smallfaces, numfeatures)
nmffaces = map(x->min(x, 1.0), nmfresult.W * nmfresult.H)
#showimgs(map(i->nmfresult.W[:, i], 1:size(nmfresult.W, 2))...)
#=
for i = 1:size(nmfresult.W, 2)
	showimg(nmfresult.W[:, i])
end
for i = rand(1:numsmall, 10)
	showimgs(smallfaces[:, i], nmffaces[:, i])
end
@show vecnorm(nmffaces - smallfaces)
nmfB = copy(nmfresult.W)
rescaleB!(nmfB)
=#
solver = ThreeQ.DWQMI.getdw2xsys4(mytoken)
#solver = ThreeQ.DWQMI.defaultsolver
num_reads = 10000
#@time B, C = Origami.factor(smallfaces, numfeatures; B=nmfB, qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, param_chain_factor=3e0)
@time B, C = Origami.factor(smallfaces, numfeatures; qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, param_chain_factor=1e0, token=mytoken)
#showimgs(map(i->B[:, i], 1:size(B, 2))...)
for i = 1:size(B, 2)
	showimg(B[:, i])
end
A = B * C
for i = rand(1:numsmall, 10)
	showimgs(smallfaces[:, i], A[:, i])
end
