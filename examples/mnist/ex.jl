import Colors
import Images
import MNIST
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
	display(Colors.Gray.(reshape(x, 28, 28))); println()
end
function showimgs(x...)
	display(Colors.Gray.(hcat(map(y->reshape(y, 28, 28), x)...))); println()
end

imgs, labels = MNIST.traindata()
imgs /= 255
numsmall = 300
smallimgs = imgs[:, 1:numsmall]
smalllabels = labels[1:size(smallimgs, 2)]
numfeatures = 30
@time nmfresult = NMF.nnmf(smallimgs, numfeatures)
nmfimgs = map(x->min(x, 1.0), nmfresult.W * nmfresult.H)
#=
showimgs(map(i->nmfresult.W[:, i], 1:size(nmfresult.W, 2))...)
for i = 1:10
	showimgs(smallimgs[:, i], nmfimgs[:, i])
end
=#
@show vecnorm(nmfimgs - smallimgs)
nmfB = copy(nmfresult.W)
rescaleB!(nmfB)
solver = ThreeQ.DWQMI.getdw2xsys4(mytoken)
#solver = ThreeQ.DWQMI.defaultsolver
num_reads = 100
#@time B, C = Origami.factor(smallimgs, numfeatures; B=nmfB, qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, param_chain_factor=3e0)
@time B, C = Origami.factor(smallimgs, numfeatures; qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, param_chain_factor=1e0)
showimgs(map(i->B[:, i], 1:size(B, 2))...)
A = B * C
for i = 1:5
	showimgs(smallimgs[:, i], A[:, i])
end
