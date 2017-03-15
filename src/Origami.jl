module Origami

import JuMP
import Ipopt
import ProgressMeter
import RobustPmap
import ThreeQ

#the basic setup here is A = B*C
#where A is a real-valued matrix of observations, B is an unknown real-valued matrix, and C is an unknown binary matrix

function finishsolve_helper(C, j, Qmat, embeddedanswer, embeddings)
	answer = 0.5 * (ThreeQ.DWQMI.unembedanswer(embeddedanswer["solutions"], embeddings)' + 1)
	besti = 1
	bestenergy = Inf
	for i = 2:size(answer, 2)
		thisenergy = ThreeQ.evalqubo(Qmat, answer[:, i])
		if thisenergy < bestenergy
			besti = i
			bestenergy = thisenergy
		end
	end
	C[:, j] = answer[:, besti]
end

"""
```
solvequbo(A, B)
```
Returns a binary matrix C that makes ||A - B*C|| small.
"""
function solvequbo(A, B, qubosolver; timeout=size(A, 2) * 3, kwargs...)
	Qs = Any[]
	stuffs = Any[]
	C = SharedArray(Float64, size(B, 2), size(A, 2))
	for j = 1:size(A, 2)
		Q = setupsmallqubo(A, B, j)
		push!(Qs, Q)
		stuff = ThreeQ.solvesapi!(Q; async=true, solver=qubosolver, reuse_embedding=true, auto_scale=true, kwargs...)
		push!(stuffs, stuff)
	end
	ps = map(s->s[1], stuffs)
	embeddingss = map(s->s[2], stuffs)
	i2varstrings = map(s->s[3], stuffs)
	ThreeQ.await_finishsolve!(Qs, ps, embeddingss, i2varstrings; finishsolve_helper=(i, Qmat, ea, emb)->finishsolve_helper(C, i, Qmat, ea, emb), kwargs...)
	return C
end

function setupsmallqubo(A, B, j)
	Q = zeros(size(B, 2), size(B, 2))
	for k = 1:size(B, 2)
		for i = 1:size(A, 1)
			Q[k, k] += B[i, k] * (B[i, k] - 2 * A[i, j])
		end
		for l = 1:k - 1
			for i = 1:size(A, 1)
				Q[k, l] += 2 * B[i, k] * B[i, l]
			end
		end
	end
	return Q
end

"""
```
solvelsq(A, C)
```
Returns a real-valued matrix B that makes ||A - B*C|| small.
"""
function solvelsq(A, C; max_iter=100, print_level=0, regularization=1e-2)
	Brows = RobustPmap.rpmap(i->solvesmalllsq(A, C, i; max_iter=max_iter, print_level=print_level, regularization=regularization), 1:size(A, 1))
	return hcat(Brows...)'
end

function solvesmalllsq(A, C, i; max_iter=100, print_level=0, regularization=1e-2)
	#A[i, j] = B[i, :]⋅C[:, j]
	m = JuMP.Model(solver=Ipopt.IpoptSolver(max_iter=max_iter, print_level=print_level))
	Browi0 = map(x->max(0.0, x), At_ldiv_B(C, A[i, :]))
	@JuMP.variable(m, Browi[j=1:size(C, 1)], start=Browi0[j])
	@JuMP.constraint(m, Browi .>= 0)
	@JuMP.objective(m, Min, sum((A[i, j] - sum(Browi[k] * C[k, j] for k = 1:size(C, 1))) ^ 2 for j = 1:size(A, 2)) + regularization * sum(Browi[k] ^ 2 for k = 1:size(C, 1)))
	JuMP.solve(m)
	return JuMP.getvalue(Browi)
end

function factor(A, k; B=rand(size(A, 1), k), C=rand([0, 1], k, size(A, 2)), min_iter=3, max_iter=100, max_lsq_iter=100, print_level=0, tol=1e-6, tol_progress=1e-6, regularization=1e-2, qubosolver=ThreeQ.DWQMI.defaultsolver, callback=(B,C,i,tlsq,tqubo)->nothing, kwargs...)
	bestB = B
	bestC = C
	lastnorm = Inf
	bestnorm = Inf
	tlsq = 0.0
	tqubo = 0.0
	callback(B, C, 0, tlsq, tqubo)
	for i = 1:max_iter
		tqubo += @elapsed C = solvequbo(A, B, qubosolver; kwargs...)
		tlsq += @elapsed B = solvelsq(A, C; max_iter=max_lsq_iter, print_level=print_level, regularization=regularization)
		callback(B, C, i, tlsq, tqubo)
		thisnorm = vecnorm(A - B * C)
		if thisnorm < bestnorm
			bestB = B
			bestC = C
			bestnorm = thisnorm
		end
		println("relative error: $(thisnorm / vecnorm(A))")
		if thisnorm < tol || thisnorm > lastnorm - tol_progress
			if i > min_iter
				break
			end
		end
		lastnorm = thisnorm
	end
	thisnorm = vecnorm(A - bestB * bestC)
	println("relative error: $(thisnorm / vecnorm(A))")
	@show tlsq, tqubo
	return bestB, bestC
end

end
