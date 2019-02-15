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
numsmall = size(faces, 2)
smallfaces = faces[:, 1:numsmall]
numfeatures = 35

#=
@time nmfresult = NMF.nnmf(smallfaces, numfeatures)
nmffaces = map(x->min(x, 1.0), nmfresult.W * nmfresult.H)
@show vecnorm(nmffaces - smallfaces) / vecnorm(smallfaces)
nmfB = copy(nmfresult.W)
rescaleB!(nmfB)
=#

const tqubos = Float64[]
const tlsqs = Float64[]
function callback(B, C, i, tlsq, tqubo)
	push!(tlsqs, tlsq)
	push!(tqubos, tqubo)
	@show tlsqs
	@show tqubos
	JLD.save("BnC_iteration_$(i)_$(numsmall)_$(numfeatures)_$(num_reads).jld", "B", B, "C", C, "tqubos", tqubos, "tlsqs", tlsqs)
end
if !(@isdefined solver)
	solver = ThreeQ.DWQMI.getdw2q(mytoken)
	#solver = ThreeQ.DWQMI.defaultsolver
	adjacency = ThreeQ.DWQMI.getadjacency(solver)
end
num_reads = 1000
#=
for i = 1:10
	reload("ThreeQ"); reload("Origami")
	try
		=#
		@time B, C = Origami.factor(smallfaces, numfeatures; qubosolver=solver, num_reads=num_reads, timeout=num_reads * numsmall / 1000 * 3 + 60, min_iter=3, token=mytoken, adjacency=adjacency, callback=callback, embedding_dir=abspath("embeddings"))
		#JLD.save("BnC_$(numsmall)_$(numfeatures)_$(num_reads)_goodalgofixed.jld", "B", B, "C", C)
		#JLD.save("BnC_$(numsmall)_$(numfeatures)_$(num_reads)_$i.jld", "B", B, "C", C)
		#=
	catch
		warn("run $i failed")
		nothing
	end
end
=#
