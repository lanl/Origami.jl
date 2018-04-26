Origami: Nonnegative/binary matrix factorization with a D-Wave quantum annealer
===============================

Description
-----------

Origami is a [Julia](http://julialang.org/) module that factors a matrix into the product of two low-rank matrices. One of the matrices has nonnegative components and the other has binary components.

A number of [examples](https://github.com/lanl/Origami.jl/tree/master/examples) are including that illustrate how to use Origami. A manuscript describing the methodology can be read [here](https://arxiv.org/abs/1704.01605).

Installation
------------

Origami can be installed by running `Pkg.clone("https://github.com/lanl/Origami.jl.git")` from within Julia. [ThreeQ](https://github.com/lanl/ThreeQ.jl) is also required to use Origami.

License
-------

Origami is provided under a BSD-ish license with a "modifications must be indicated" clause.  See LICENSE.md file for the full text.

This package is part of the Hybrid Quantum-Classical Computing suite, known internally as LA-CC-16-032.

Author
------

Daniel O'Malley, <omalled@lanl.gov>
