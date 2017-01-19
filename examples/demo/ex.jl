import Origami

srand(1)
B = rand(2, 3)
C = rand([0, 1], 3, 2)
A = B * C
Cest = Origami.solvequbo(A, B)
