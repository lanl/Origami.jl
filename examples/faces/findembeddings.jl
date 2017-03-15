import ThreeQ

function getcompleteQ(n)
	Q = Dict()
	for i = 0:n - 1
		for j = 0:i
			Q[(i, j)] = 1.0
		end
	end
	return Q
end

solver = ThreeQ.DWQMI.getdw2xsys4(mytoken)
adjacency = ThreeQ.DWQMI.getadjacency(solver)
bestembeddings = Dict()
for n = 30:40
	@show n
	numoutertries = 100
	bestmaxchain = Inf
	for i = 1:numoutertries
		embeddings = ThreeQ.findembeddings(getcompleteQ(n), adjacency, false; timeout=60*10, verbose=0, tries=1000, embedding_dir=abspath("embeddings"))
		if !haskey(bestembeddings, n)
			bestembeddings[n] = embeddings
		elseif maximum(map(length, bestembeddings[n])) > maximum(map(length, embeddings))
			println("replacing")
			@show maximum(map(length, bestembeddings[n])), maximum(map(length, embeddings))
			bestembeddings[n] = embeddings
		end
	end
end
