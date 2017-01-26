module Origami

import ThreeQ
import JuMP
import Ipopt

#the basic setup here is A = B*C
#where A is a real-valued matrix of observations, B is an unknown real-valued matrix, and C is an unknown binary matrix

"""
```
solvequbo(A, B)
```
Returns a binary matrix C that makes ||A - B*C|| small.
"""
function solvequbo(A, B)
	C = Array(Int, size(B, 2), size(A, 2))
	#decompose the problem into one QUBO for each column of C
	for j = 1:size(A, 2)
		m = ThreeQ.Model("Origami", "laptop", "c4-sw_sample", "workingdir", "c4")
		@ThreeQ.defvar m Ccolj[1:size(B, 2)]
		for i = 1:size(A, 1)
			#A[i, j] == B[i, :]⋅C[:, j]
			ThreeQ.addlinearconstraint!(m, A[i, j], B[i, :], [Ccolj[k] for k in 1:size(B, 2)])
		end
		ThreeQ.qbsolv!(m; showoutput=false)
		C[:, j] = Ccolj.value
	end
	return C
end

"""
```
solvelsq(A, C)
```
Returns a real-valued matrix B that makes ||A - B*C|| small.
"""
function solvelsq(A, C; max_iter=100, print_level=0)
	m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=max_iter, print_level=print_level))
	@JuMP.variable(m, B[i=1:size(A, 1), j=1:size(C, 1)])
	@JuMP.constraint(m, B .>= 0)
	@JuMP.objective(m, Min, sum((A[i, j] - sum(B[i, k] * C[k, j] for k = 1:size(C, 1))) ^ 2 for i = 1:size(A, 1), j = 1:size(A, 2)))
	JuMP.solve(m)
	return JuMP.getvalue(B)
end

function factor(A, k; B=rand(size(A, 1), k), C=rand([0, 1], k, size(A, 2)), max_iter=100, max_lsq_iter=100, print_level=0, tol=1e-6)
	for i = 1:max_iter
		C = solvequbo(A, B)
		B = solvelsq(A, C; max_iter=max_lsq_iter, print_level=print_level)
		if vecnorm(A - B * C) < tol
			break
		end
	end
	return B, C
end

end
