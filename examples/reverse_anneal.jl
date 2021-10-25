import JLD
import JuMP
import Ipopt
import LinearAlgebra
import RobustPmap
import SharedArrays
using Statistics
using StatsBase

"""
Origami code
"""

function solvelsq(A, C; max_iter=100, print_level=0, regularization=1e-2)
	Brows = RobustPmap.rpmap(i->solvesmalllsq(A, C, i; max_iter=max_iter, print_level=print_level, regularization=regularization), 1:size(A, 1))
	return hcat(Brows...)'
end

function solvesmalllsq(A, C, i; max_iter=100, print_level=0, regularization=1e-2)
	m = JuMP.Model(JuMP.with_optimizer(Ipopt.Optimizer; max_iter=max_iter, print_level=print_level))
	Browi0 = map(x->max(0.0, x), C' \ A[i, :])
	@JuMP.variable(m, Browi[j=1:size(C, 1)], start=Browi0[j])
	@JuMP.constraint(m, Browi .>= 0)
	@JuMP.objective(m, Min, sum((A[i, j] - sum(Browi[k] * C[k, j] for k = 1:size(C, 1))) ^ 2 for j = 1:size(A, 2)) + regularization * sum(Browi[k] ^ 2 for k = 1:size(C, 1)))
	JuMP.optimize!(m)
	return JuMP.value.(Browi)
end

function solveQUBOs(A, B, oldC, solver; reverse_anneal=false, temp=0.2, kwargs...)
	Qs = Any[]
	callsToDWave = Any[]
	C = SharedArrays.SharedArray{Float64}(size(B, 2), size(A, 2))
	for j = 1:size(A, 2)
		Q = setupsmallqubo(A, B, j)
		push!(Qs, Q)
		if !reverse_anneal
			callToDWave = DWave(Q, embedding)
		else
			anneal_schedule = annealSchedule(temp, 10)
			initStateUnembedded = initStateFromEmbedding(embedding, oldC[:,j])
			callToDWave = DWave(Q, embedding; initial_state=initStateUnembedded, anneal_schedule=anneal_schedule)
		end
		push!(callsToDWave, callToDWave)
	end
	ps = map(s->s[1], callsToDWave)
	embeddingss = map(s->s[2], callsToDWave)
	i2varstrings = map(s->s[3], callsToDWave)
	unembed_answers(Qs, ps, embeddingss, i2varstrings; finishsolve_helper=(i, Qmat, ea, emb)->finishsolve_helper(C, i, Qmat, ea, emb))
	return C
end

function finishsolve_helper(C, j, Qmat, embeddedanswer, embeddings)
	answer = [Int(x) for x in (0.5 * (DWave.unembedanswer(embeddedanswer["solutions"], embeddings)' .+ 1))]
	besti = 1
	bestenergy = Inf
	for i = 2:size(answer, 2)
		thisenergy = evalqubo(Qmat, answer[:, i])
		if thisenergy < bestenergy
			besti = i
			bestenergy = thisenergy
		end
	end
	C[:, j] = answer[:, besti]
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

function factor(A, k; B=rand(size(A, 1), k), C=rand([0, 1], k, size(A, 2)), reverse_anneal=false, temp=0.2, iters=10, callback=(B,C,i)->nothing, kwargs...)
	bestB = B
	bestC = C
	lastnorm = Inf
	bestnorm = Inf
	callback(B,C,0)
	for i = 1:iters
#		println("solving least squares")
		B = solvelsq(A, C)
		C = solveQUBOs(A, B, C, solver; reverse_anneal=reverse_anneal, temp=temp, kwargs...)
		callback(B,C,i)
		thisnorm = LinearAlgebra.norm(A - B * C)
		if thisnorm < bestnorm
			bestB = B
			bestC = C
			bestnorm = thisnorm
		end
		println("finished iter $(i) with relative error: $(thisnorm / LinearAlgebra.norm(A))")
		lastnorm = thisnorm

	end
	return bestB, bestC
end

"""
Reverse anneal code
"""

function forwardAnneal(embedding, Q; auto_scale=true, num_reads=1000)
    len=size(Q)[1]

    adjacency = DWave.getadjacency(solver)

    solutions = []
    energies = []
    function finishsolve_helper(m, embans, emb)
        solutions = [Int(x) for x in (0.5 * (DWave.unembedanswer(embans["solutions"], emb)' .+ 1))]
        energies = embans["energies"]
    end
    DWave(Q; auto_scale=auto_scale, solver=solver, num_reads=num_reads, embeddings=embedding, finishsolve_helper=finishsolve_helper, token=mytoken, answer_mode="raw")
    return histogramForDW(solutions, energies)
end

function initStateFromEmbedding(embedding, initState)
	embeddingP1 = [x .+ 1 for x in embedding]
	initStateUnembedded = []
	for _ in 1:2048
		push!(initStateUnembedded, 3)
	end
	for i in 1:length(embeddingP1)
		for bit in embeddingP1[i]
			initStateUnembedded[bit] = Int(2*(initState[i]-1/2))
		end
	end
	return initStateUnembedded
end

function reverseAnneal(embedding, Q, initial_state, anneal_schedule; auto_scale=true, num_reads=1000)
	initial_state_unembedded = initStateFromEmbedding(embedding, initial_state)
	len=size(Q)[1]
    adjacency = DWave.getadjacency()
    solutions = []
    energies = []
    function finishsolve_helper(m, embans, emb)
        solutions = [Int(x) for x in (0.5 * (DWave.unembedanswer(embans["solutions"], emb)' .+ 1))]
        energies = embans["energies"]
    end
    DWave(Q; anneal_schedule = anneal_schedule, initial_state = initial_state_unembedded, auto_scale=auto_scale, solver=solver, num_reads=num_reads, embeddings=embedding, finishsolve_helper=finishsolve_helper, token=mytoken, answer_mode="raw")
    return histogramForDW(solutions, energies)
end

function annealSchedule(s, t)
    # s is the "temperature" (unitless, between 0 and 1)
    # t is the time spent at s (microseconds)
    # always do a 10microsec warm up and then 10microsec cooldown
    res = [(0,1.0)]
    push!(res, (10,1-s))
    push!(res, (10+t, 1-s))
    push!(res, (20+t, 1))
    return res
end

function testTemp(embedding, Q, bestForwardSolution, temp)
	reverseSolutions = reverseAnneal(embedding, Q, bestForwardSolution, annealSchedule(temp,10))

	bestForwardSolutionEnergy = evalqubo(Q, bestForwardSolution)

	better = 0
	same = 0
	worse = 0

	for solution in reverseSolutions
		if evalqubo(Q, solution[3]) < bestForwardSolutionEnergy
			better += solution[2]
		elseif evalqubo(Q, solution[3]) == bestForwardSolutionEnergy
			same += solution[2]
		else
			worse += solution[2]
		end
	end

	return (better, same, worse)
end

function evalqubo(Qmat, solution)
	energy = 0.0
	for i = 1:size(Qmat, 1)
		if solution[i] == 1
			energy += Qmat[i, i]
			for j = 1:i - 1
				energy += (Qmat[i, j] + Qmat[j, i]) * solution[j]
			end
		end
	end
	return energy
end