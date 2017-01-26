using Base.Test
import Origami

srand(1)
B = rand(2, 3)
C = rand([0, 1], 3, 2)
A = B * C
Cest = Origami.solvequbo(A, B)
@test_approx_eq_eps A B * Cest 1e-6
Best = Origami.solvelsq(A, C)
@test_approx_eq_eps A Best * C 1e-6

Best, Cest = Origami.factor(A, 3; tol=1e-6)
#vecnorm(A - Best2 * Cest2)
@test_approx_eq_eps A Best * Cest 1e-6
