import Colors
import FileIO
import Glob
#import Gurobi
import Images
import JLD
import JuMP
import NMF
import Origami
import PyPlot
import ThreeQ

const imsize = 19
const border = 2

function doubleit(x)
	y = similar(x, 2 * size(x, 1), 2 * size(x, 2))
	for i = 1:size(x, 1), j = 1:size(x, 2)
		y[2 * i - 1, 2 * j - 1] = x[i, j]
		y[2 * i, 2 * j - 1] = x[i, j]
		y[2 * i - 1, 2 * j] = x[i, j]
		y[2 * i, 2 * j] = x[i, j]
	end
	return y
end

function showimgs(x...)
	display(Colors.Gray.(hcat(map(y->doubleit(doubleit(reshape(y, imsize, imsize))), x)...))); println()
end

function truesyn(A, B, C, i)
	x = fill(Colors.RGB(1, 1, 1), (imsize + 2 * border) * 5, (imsize + 2 * border))
	x[1 * (imsize + 2 * border) + 1:1 * (imsize + 2 * border) + imsize, border:border+imsize - 1] = Colors.Gray.(reshape(A[:, i], imsize, imsize))
	x[3 * (imsize + 2 * border) + 1:3 * (imsize + 2 * border) + imsize, border:border+imsize - 1] = Colors.Gray.(reshape(map(x->min(1.0, x), B * C[:, i]), imsize, imsize))
	return x
end

function breakdownimg(A, B, C, i, color, x=fill(Colors.RGB(0, 0, 0), (imsize + 2 * border) * 5, (imsize + 2 * border) * 7); scale=false)
	for j = 1:size(C, 1)
		row = div(j - 1, 7) + 1
		col = mod(j - 1, 7) + 1
		subimagei0 = (row - 1) * (imsize + 2 * border) + 1
		subimagei1 = row * (imsize + 2 * border)
		subimagej0 = (col - 1) * (imsize + 2 * border) + 1
		subimagej1 = col * (imsize + 2 * border)
		if C[j, i] == 1
			x[subimagei0:subimagei1, subimagej0:subimagej1] = color
		else
			x[subimagei0:subimagei1, subimagej0:subimagej1] = Colors.RGB(1, 1, 1)
		end
		if scale
			x[subimagei0+border:subimagei1-border, subimagej0+border:subimagej1-border] = Colors.Gray.(reshape((B[:, j] - minimum(B[:, j])) / (maximum(B[:, j]) - minimum(B[:, j])), imsize, imsize))
		else
			x[subimagei0+border:subimagei1-border, subimagej0+border:subimagej1-border] = Colors.Gray.(reshape(B[:, j], imsize, imsize))
		end
	end
	return x
end

function doface(A, B, C, i, color, doubles)
	rightimg = breakdownimg(A, B, C, i, color)
	leftimg = truesyn(A, B, C, i)
	wholeimg = hcat(leftimg, fill(Colors.RGB(1, 1, 1), 5 * (imsize + 2 * border), 5), rightimg)
	for i = 1:doubles
		wholeimg = doubleit(wholeimg)
	end
	return wholeimg
end

dirs = ["run1_40bits_crashed", "run2_40bits_finished", "run3_35bits_crashed", "run4_35bits_crashed", "run5_35bits_finished", "run6_35bits_10samples_crashed", "run7_35bits_crashed", "run8_35bits_crashed"]

A = JLD.load("faces.jld", "faces")
B, C = JLD.load("run8_35bits_crashed/BnC_iteration_8_2429_35_10000.jld", "B", "C")

#=
numselected = 3
m = JuMP.Model(solver=Gurobi.GurobiSolver())
@JuMP.variable(m, x[1:size(C, 2)], Bin)
for i = 1:size(C, 1)
	@JuMP.constraint(m, sum(C[i, j] * x[j] for j = 1:size(C, 2)) <= 1)
end
@JuMP.objective(m, Min, sum(norm(A[:, j] - B * C[:, j]) * x[j] for j = 1:size(C, 2)))
@JuMP.constraint(m, sum(x[j] for j = 1:size(C, 2)) == numselected)
@time JuMP.solve(m)
@show JuMP.getobjectivevalue(m)
xval = JuMP.getvalue(x)
is = map(x->x[1], sort(collect(zip(1:length(xval), xval)), by=x->-x[2])[1:numselected])
=#
#is = rand(1:size(C, 2), 3)
#is = [indmin(map(i->norm(A[:, i] - B * C[:, i]) / norm(A[:, i]), 1:size(A, 2)))]
is = [1, indmin(map(i->norm(A[:, i] - B * C[:, i]) / norm(A[:, i]), 1:size(A, 2)))]
@show is
#=
for i = 1:10:size(A, 2)
	@show i
	showimgs(map(j->A[:, j], i:i + 9)...)
end
=#
for i in is
	parts = []
	for j = 1:size(C, 1)
		if C[j, i] == 1
			push!(parts, B[:, j])
		end
	end
	face = doface(A, B, C, i, Colors.RGB(141 / 255, 206 / 255, 69 / 255), 2)
	FileIO.save("face_$i.tif", doubleit(doubleit(face)))
	display(face); println()
	#=
	showimgs(A[:, i], B * C[:, i], map(x->0 + x, parts)...)
	display(doubleit(doubleit(breakdownimg(A, B, C, i, Colors.RGB(0, 1, 0))))); println()
	display(doubleit(doubleit(breakdownimg(A, B, C, i, Colors.RGB(0, 1, 0); scale=true)))); println()
	=#
	contrasted = doubleit(doubleit(doubleit(doubleit(breakdownimg(A, B, C, i, Colors.RGB(1, 1, 1); scale=true)))))
	FileIO.save("contrasted_features.tif", contrasted)
	display(contrasted); println()
end
#display(doubleit(doubleit(breakdownimg(A, B, C, 1, Colors.RGB(0, 1, 0))))); println()

