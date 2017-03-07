module Origami

import JuMP
import Ipopt
import ProgressMeter
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
function solvequbo(A, B, qubosolver::Qbsolv; timeout=size(A, 2) * 3, kwargs...)
	C = Array(Float64, size(B, 2), size(A, 2))
	badcols = Array{Int64, 1}[]
	goodcols = Array{Int64, 1}[]
	for j = 1:size(A, 2)
		m, Ccol = setupsmallqubo(A, B, j; connection="online", solver="DW2X_SYS4")
		ThreeQ.qbsolv!(m; minval=-sum(A[:, j] .^ 2))
		C[:, j] = Ccol.value
	end
	return C
end

function solvequbo(A, B, qubosolver; timeout=size(A, 2) * 3, kwargs...)
	ms = Any[]
	Ccols = Any[]
	stuffs = Any[]
	C = Array(Float64, size(B, 2), size(A, 2))
	ready = fill(false, size(A, 2))
	submittimes = zeros(Float64, 5)
	ThreeQ.DWQMI.zerototaltimes!()
	ThreeQ.zerototaltimes!()
	@ProgressMeter.showprogress 1 "Submit QUBOs..." for j = 1:size(A, 2)
		submittimes[1] += @elapsed m, Ccol = setupsmallqubo(A, B, j)
		submittimes[2] += @elapsed push!(ms, m)
		submittimes[3] += @elapsed push!(Ccols, Ccol)
		submittimes[4] += @elapsed stuff = ThreeQ.solvesapi!(m; async=true, solver=qubosolver, reuse_embedding=true, auto_scale=true, kwargs...)
		ready[j] = true
		submittimes[5] += @elapsed push!(stuffs, stuff)
	end
	@show submittimes
	@show ThreeQ.DWQMI.totaltimes
	@show ThreeQ.totaltimes
	numinvalid = 0
	t0 = now()
	timeawaitcomplete = t0 - t0
	ps = map(s->s[1], stuffs)
	embeddingss = map(s->s[2], stuffs)
	i2varstrings = map(s->s[3], stuffs)
	println("await_finish")
	@time ThreeQ.await_finishsolve!(ms, ps, embeddingss, i2varstrings; kwargs...)
	@ProgressMeter.showprogress 1 "Get results..." for j = 1:size(A, 2)
		#=
		while !ready[j]
			sleep(0.01)
		end
		t0 = now()
		if !ThreeQ.DWQMI.dwcore.await_completion([stuffs[j][1]], 1, 60)
			error("timed out waiting for a problem to complete")
		end
		timeawaitcomplete += now() - t0
		=#
		m = ms[j]
		stuff = stuffs[j]
		#ThreeQ.finishsolve!(m, stuff...; kwargs...)
		besti = 1
		bestenergy = Inf
		gotavalid = false
		for i = 1:length(m.energies)
			gotavalid = gotavalid || m.valid[i]
			if m.valid[i] && m.energies[i] < bestenergy
				bestenergy = m.energies[i]
				besti = i
			end
		end
		if gotavalid == false
			numinvalid += 1
			for i = 1:length(m.energies)
				if m.energies[i] < bestenergy
					bestenergy = m.energies[i]
					besti = i
				end
			end
		end
		@ThreeQ.loadsolution m energy occurrences isvalid besti
		C[:, j] = Ccols[j].value
	end
	println("total waiting time: $(float(timeawaitcomplete) / 1000) seconds")
	if numinvalid > 0
		warn("$(numinvalid / size(A, 2) * 100)% have no valid solutions")
	end
	return C
end

function setupsmallqubonew(A, B, j)
	Q = zeros(size(B, 2), size(B, 2))
	for k = 1:size(B, 2)
		for i = 1:size(A, 1)
			Q[k] += B[i, k] * (B[i, k] - 2 * A[i, j])
		end
		for l = 1:k - 1
			for i = 1:size(A, 1)
				Q[k, l] += 2 * B[i, k] * B[i, l]
			end
		end
	end
	return Q
end

function setupsmallqubo(A, B, j; connection="lanl", solver="DW2X", workspace="lanl_dw2x")
	m = ThreeQ.Model("Origami_$j", connection, solver, "workingdir", workspace)
	@ThreeQ.defvar m Ccolj[1:size(B, 2)]
	for k = 1:size(B, 2)
		lincoeff = 0.0
		for i = 1:size(A, 1)
			lincoeff += B[i, k] * (B[i, k] - 2 * A[i, j])
		end
		@ThreeQ.addterm m lincoeff * Ccolj[k]
		for l = 1:k - 1
			quadcoeff = 0.0
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

function factor(A, k; B=rand(size(A, 1), k), C=rand([0, 1], k, size(A, 2)), min_iter=3, max_iter=100, max_lsq_iter=100, print_level=0, tol=1e-6, tol_progress=1e-6, regularization=1e-2, qubosolver=Qbsolv(), callback=(B,C,i)->nothing, kwargs...)
	bestB = B
	bestC = C
	lastnorm = Inf
	bestnorm = Inf
	tlsq = 0.0
	tqubo = 0.0
	callback(B, C, 0)
	for i = 1:max_iter
		tqubo += @elapsed C = solvequbo(A, B, qubosolver; kwargs...)
		tlsq += @elapsed B = solvelsq(A, C; max_iter=max_lsq_iter, print_level=print_level, regularization=regularization)
		callback(B, C, i)
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
