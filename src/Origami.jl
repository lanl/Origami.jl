module Origami

import JuMP
import Ipopt
import RobustPmap
import ThreeQ

type Qbsolv end

#the basic setup here is A = B*C
#where A is a real-valued matrix of observations, B is an unknown real-valued matrix, and C is an unknown binary matrix

"""
```
solvequbo(A, B)
```
Returns a binary matrix C that makes ||A - B*C|| small.
"""
function solvequbo(A, B, qubosolver=Qbsolv(); timeout=size(A, 2) * 3, kwargs...)
	ms = Any[]
	Ccols = Any[]
	stuffs = Any[]
	C = Array(Float64, size(B, 2), size(A, 2))
	println("submit")
	@time for j = 1:size(A, 2)
		m, Ccol = setupsmallqubo(A, B, j)
		push!(ms, m)
		push!(Ccols, Ccol)
		stuff = ThreeQ.solvesapi!(m; async=true, solver=qubosolver, reuse_embedding=true, auto_scale=true, kwargs...)
		push!(stuffs, stuff)
	end
	println("await")
	@time ThreeQ.DWQMI.dwcore.await_completion(map(x->x[1], stuffs), length(stuffs), timeout)
	println("finish")
	@time for j = 1:size(A, 2)
		m = ms[j]
		stuff = stuffs[j]
		ThreeQ.finishsolve!(m, stuff...)
		besti = 1
		for i = 1:length(m.energies)
			@ThreeQ.loadsolution m energy occurrences isvalid i
			if isvalid && energy < m.energies[besti]
				besti = i
			end
		end
		@ThreeQ.loadsolution m energy occurrences isvalid besti
		C[:, j] = Ccols[j].value
	end
	return C
end

function setupsmallqubo(A, B, j)
	m = ThreeQ.Model("Origami_$j", "lanl", "DW2X", "workingdir", "lanl_dw2x")
	@ThreeQ.defvar m Ccolj[1:size(B, 2)]
	for k = 1:size(B, 2)
		lincoeff = 0.
		for i = 1:size(A, 1)
			lincoeff += B[i, k] * (B[i, k] - 2 * A[i, j])
		end
		@ThreeQ.addterm m lincoeff * Ccolj[k]
		quadcoeff = 0.
		for l = 1:k - 1
			for i = 1:size(A, 1)
				quadcoeff += 2 * B[i, k] * B[i, l]
			end
			@ThreeQ.addterm m quadcoeff * Ccolj[k] * Ccolj[l]
		end
	end
	return m, Ccolj
end

function solvequboold(A, B, qubosolver=Qbsolv(); kwargs...)
	Ccols = RobustPmap.rpmap(j->solvesmallqubo(A, B, j, qubosolver; kwargs...), 1:size(A, 2))
	#Ccols = map(j->solvesmallqubo(A, B, j, qubosolver; kwargs...), 1:size(A, 2))
	return hcat(Ccols...)
end

@generated function solvesmallqubo(A, B, j, qubosolver; kwargs...)
	if typeof(qubosolver) == Qbsolv
		println("qbsolv")
		solvecode = :(ThreeQ.qbsolv!(m; showoutput=false))
	else
		println("not qbsolv")
		solvecode = quote
			ThreeQ.solvesapi!(m; solver=qubosolver, auto_scale=true, kwargs...)
			#=
			kwargdict = Dict(zip(map(x->x[1], kwargs), map(x->x[2], kwargs)))
			ThreeQ.solve!(m; removefiles=true, numreads=kwargdict[:num_reads], showoutput=false)
			=#
			besti = 1
			for i = 1:length(m.energies)
				@ThreeQ.loadsolution m energy occurrences isvalid i
				if isvalid && energy < m.energies[besti]
					besti = i
				end
			end
			@ThreeQ.loadsolution m energy occurrences isvalid besti
		end
	end
	q = quote
		#m = ThreeQ.Model("Origami_$j", "laptop", "c4-sw_sample", "workingdir", "c4")
		m = ThreeQ.Model("Origami_$j", "lanl", "DW2X", "workingdir", "lanl_dw2x")
		@ThreeQ.defvar m Ccolj[1:size(B, 2)]
		for i = 1:size(A, 1)
			#A[i, j] == B[i, :]⋅C[:, j]
			ThreeQ.addlinearconstraint!(m, A[i, j], B[i, :], [Ccolj[k] for k in 1:size(B, 2)])
		end
		$solvecode
		return Ccolj.value
	end
	return q
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
	@JuMP.variable(m, Browi[1:size(C, 1)])
	@JuMP.constraint(m, Browi .>= 0)
	@JuMP.objective(m, Min, sum((A[i, j] - sum(Browi[k] * C[k, j] for k = 1:size(C, 1))) ^ 2 for j = 1:size(A, 2)) + regularization * sum(Browi[k] ^ 2 for k = 1:size(C, 1)))
	JuMP.solve(m)
	return JuMP.getvalue(Browi)
end

function factor(A, k; B=rand(size(A, 1), k), C=rand([0, 1], k, size(A, 2)), min_iter=3, max_iter=100, max_lsq_iter=100, print_level=0, tol=1e-6, tol_progress=1e-6, regularization=1e-2, qubosolver=Qbsolv(), kwargs...)
	bestB = B
	bestC = C
	lastnorm = Inf
	bestnorm = Inf
	tlsq = 0.0
	tqubo = 0.0
	Main.showimgs(map(i->B[:, i], 1:size(B, 2))...)
	for i = 1:max_iter
		tqubo += @elapsed C = solvequbo(A, B, qubosolver; kwargs...)
		tlsq += @elapsed B = solvelsq(A, C; max_iter=max_lsq_iter, print_level=print_level, regularization=regularization)
		Main.showimgs(map(i->B[:, i], 1:size(B, 2))...)
		thisnorm = vecnorm(A - B * C)
		if thisnorm < bestnorm
			bestB = B
			bestC = C
			bestnorm = thisnorm
		end
		@show thisnorm
		if thisnorm < tol || thisnorm > lastnorm - tol_progress
			if i > min_iter
				break
			end
		end
		lastnorm = thisnorm
	end
	@show vecnorm(A - bestB * bestC)
	@show tlsq, tqubo
	return bestB, bestC
end

end
