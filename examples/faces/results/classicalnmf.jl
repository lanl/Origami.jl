import Colors
import Images
import JLD
import NMF
import Origami
import ThreeQ

faces = JLD.load("faces.jld", "faces")
B, C = JLD.load("run8_35bits_crashed/BnC_iteration_8_2429_35_10000.jld", "B", "C")
numsmall = size(faces, 2)
smallfaces = faces[:, 1:numsmall]
numfeatures = 35

@time nmfresult = NMF.nnmf(smallfaces, numfeatures)
nmfB = copy(nmfresult.W)
nmfC = copy(nmfresult.H)
@show sum(nmfC .< 1e-8) / length(nmfC)
@show sum(C .< 1e-8) / length(C)
@show sum(nmfB .< 1e-8) / length(nmfB)
@show sum(B .< 1e-8) / length(B)
@show vecnorm(faces - B * C)
@show vecnorm(faces - nmfB * nmfC)
