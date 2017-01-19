module Origami

using ThreeQ

#the basic setup here is A = B*C
#where A is a real-valued matrix of observations, B is an unknown real-valued matrix, and C is an unknown binary matrix

"""
```
solvequbo(A, B)
```
Solves returns a binary matrix C that makes ||A - B*C|| small.
"""
function solvequbo(A, B)
	m = ThreeQ.Model("Origami", "laptop", "c4-sw_sample", "workingdir", "c4")
	@defvar m C[1:size(B, 2), 1:size(A, 2)]
	#TODO you can decompose the problem into one QUBO for each column of C
	for i = 1:size(A, 1), j = 1:size(A, 2)
		#A[i, j] == B[i, :]⋅C[:, j]
		ThreeQ.addlinearconstraint!(m, A[i, j], B[i, :], [C[k, j] for k in 1:size(C, 1)])
	end
	ThreeQ.qbsolv!(m; showoutput=false)
	return C.value
end

end
