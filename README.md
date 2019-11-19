CRAN Task View: Optimization and Mathematical Programming
---------------------------------------------------------

|                 |                                                            |
|-----------------|------------------------------------------------------------|
| **Maintainer:** | Stefan Theussl, Florian Schwendinger, Hans W. Borchers     |
| **Contact:**    | R-optimization at mailbox.org                              |
| **Version:**    | 2019-11-22                                                 |
| **URL:**        | <https://CRAN.R-project.org/view=Optimization>             |

This CRAN task view contains a list of packages which offer facilities
for solving optimization problems. Although every regression model in
statistics solves an optimization problem they are not part of this
view. If you are looking for regression methods, the following views
will contain useful starting points:
[Multivariate](https://cran.r-project.org/web/views/Multivariate.html),
[SocialSciences](https://cran.r-project.org/web/views/SocialSciences.html),
[Robust](https://cran.r-project.org/web/views/Robust.html), among others.

The focus of this task view is on [Optimization Infrastructure
Packages](#optimization-infrastructure-packages) , [General Purpose
Continuous Solvers](#general-purpose-continuous-solvers) , [Mathematical
Programming Solvers](#mathematical-programming-solvers) , and [Specific
Applications in Optimization](#specific-applications-in-optimization), or
[Multi-objective Optimization](#multi-objective-optimization).

Packages are categorized according to these sections. Many packages provide
functionality for more than one of the these subjects. E.g., mixed integer
linear programming solvers typically offer standard linear programming
routines like the simplex algorithm. Therefore please read the more detailed
package descriptions to make sure which problems can be solved. 

If you think that packages are missing from the list, please let us know.

<span id="optimization-infrastructure-packages">Optimization Infrastructure Packages</span>
-------------------------------------------------------------------------------------------

-   The [optimx](https://cran.r-project.org/package=optimx) package provides a
    replacement and extension of the `optim()` function in Base R with a
    call to several function minimization codes in R in a single
    statement. These methods handle smooth, possibly box constrained
    functions of several or many parameters. Function `optimr()` in this
    package extends the `optim()` function with the same syntax but more
    'method' choices. Function `opm()` applies several solvers to a
    selected optimization task and returns a dataframe of results for
    easy comparison.

-   The R Optimization Infrastructure
    ([ROI](https://cran.r-project.org/package=ROI)) package provides a framework 
    for handling optimization problems in R. It uses an object-oriented
    approach to define and solve various optimization tasks from
    different problem classes (e.g., linear, quadratic, non-linear
    programming problems). This makes optimizationt transparent for the user
    as the corresponding workflow is abstracted from the underlying solver.
    The approach allows for easy switching between solvers and thus enhances
    comparability.
    For more information see the [ROI home page](http://roi.r-forge.r-project.org/).

-   The package [CVXR](https://cran.r-project.org/package=CVXR) provides an
    object-oriented modeling language for Disciplined Convex Programming
    (DCP). It allows the user to formulate convex optimization problems
    in a natural way following mathematical convention and DCP rules.
    The system analyzes the problem, verifies its convexity, converts it
    into a canonical form, and hands it off to an appropriate solver
    such as ECOS or SCS to obtain the solution. (CVXR is derived from
    the MATLAB toolbox CVX, developed at Stanford University, cf. [CVXR
    home page](https://cvxr.rbind.io) .)

<span id="general-purpose-continuous-solvers">General Purpose Continuous Solvers</span>
---------------------------------------------------------------------------------------

Package stats offers several general purpose optimization routines. For
one-dimensional unconstrained function optimization there is
`optimize()` which searches an interval for a minimum or maximum.
Function `optim()` provides an implementation of the
Broyden-Fletcher-Goldfarb-Shanno (BFGS) method, bounded BFGS, conjugate
gradient (CG), Nelder-Mead, and simulated annealing (SANN) optimization
methods. It utilizes gradients, if provided, for faster convergence.
Typically it is used for unconstrained optimization but includes an
option for box-constrained optimization.

Additionally, for minimizing a function subject to linear inequality
constraints, stats contains the routine `constrOptim()`. Then there is
`nlm` which is used for solving nonlinear unconstrained minimization
problems. `nlminb()` offers box-constrained optimization using the PORT
routines.

-   Package [lbfgs](https://cran.r-project.org/package=lbfgs) wraps the libBFGS C
    library by Okazaki and Morales (converted from Nocedal's L-BFGS-B
    3.0 Fortran code), interfacing both the L-BFGS and the OWL-QN
    algorithm, the latter being particularly suited for
    higher-dimensional problems.
-   [lbfgsb3](https://cran.r-project.org/package=lbfgsb3) and
    [lbfgsb3c](https://cran.r-project.org/package=lbfgsb3c) both interface
    J.Nocedal's L-BFGS-B 3.0 Fortran code, a limited memory BFGS
    minimizer, allowing bound constraints and being applicable to
    higher-dimensional problems. ('lbfgsb3c' has an 'optim'-like
    interface based on 'Rcpp'.)
-   [optimParallel](https://cran.r-project.org/package=optimParallel)
    provides a parallel version of the L-BFGS-B method of `optim()`;
    using `optimParallel()` can significantly reduce the optimization time.
-   [RcppNumerical](https://cran.r-project.org/package=RcppNumerical) is a
    collection of open source libraries for numerical computing and
    their integration with 'Rcpp'. It provides a wrapper for the L-BFGS
    algorithm, based on the LBFGS++ library (based on code of N.
    Okazaki).
-   Package [ucminf](https://cran.r-project.org/package=ucminf) implements an
    algorithm of quasi-Newton type for nonlinear unconstrained
    optimization, combining a trust region with line search approaches.
    The interface of `ucminf()` is designed for easy interchange with
    `optim()`.
-   The following packages implement optimization routines in pure R,
    for nonlinear functions with bounds constraints:
    [Rcgmin](https://cran.r-project.org/package=Rcgmin): gradient function
    minimization similar to GC; [Rvmmin](https://cran.r-project.org/package=Rvmmin):
    variable metric function minimization;
    [Rtnmin](https://cran.r-project.org/package=Rtnmin): truncated Newton function
    minimization.
-   [mize](https://cran.r-project.org/package=mize) implements optimization
    algorithms in pure R, including conjugate gradient (CG),
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) and limited memory BFGS
    (L-BFGS) methods. Most internal parameters can be set through the
    calling interface.
-   Package [dfoptim](https://cran.r-project.org/package=dfoptim), for
    derivative-free optimization procedures, contains quite efficient R
    implementations of the Nelder-Mead and Hooke-Jeeves algorithms
    (unconstrained and with bounds constraints).
-   Package [nloptr](https://cran.r-project.org/package=nloptr) provides access to
    NLopt, an LGPL licensed library of various nonlinear optimization
    algorithms. It includes local derivative-free (COBYLA, Nelder-Mead,
    Subplex) and gradient-based (e.g., BFGS) methods, and also the
    augmented Lagrangian approach for nonlinear constraints.
-   Package [alabama](https://cran.r-project.org/package=alabama) provides an
    implementations of the Augmented Lagrange Barrier minimization
    algorithm for optimizing smooth nonlinear objective functions with
    (nonlinear) equality and inequality constraints.
-   Package [Rsolnp](https://cran.r-project.org/package=Rsolnp) provides an
    implementation of the Augmented Lagrange Multiplier method for
    solving nonlinear optimization problems with equality and inequality
    constraints (based on code by Y. Ye).
-   [NlcOptim](https://cran.r-project.org/package=NlcOptim) solves nonlinear
    optimization problems with linear and nonlinear equality and
    inequality constraints, implementing a Sequential Quadratic
    Programming (SQP) method; accepts the input parameters as a
    constrained matrix.
-   In package Rdonlp2 (see the [<span
    class="Rforge">rmetrics</span>](https://R-Forge.R-project.org/projects/rmetrics/)
    project) function `donlp2()`, a wrapper for the DONLP2 solver,
    offers the minimization of smooth nonlinear functions and
    constraints. DONLP2 can be used freely for any kind of research
    purposes, otherwise it requires licensing.
-   [clue](https://cran.r-project.org/package=clue) contains the function `sumt()`
    for solving constrained optimization problems via the sequential
    unconstrained minimization technique (SUMT).
-   [BB](https://cran.r-project.org/package=BB) contains the function `spg()`
    providing a spectral projected gradient method for large scale
    optimization with simple constraints. It takes a nonlinear objective
    function as an argument as well as basic constraints.
-   [GrassmannOptim](https://cran.r-project.org/package=GrassmannOptim) is a package
    for Grassmann manifold optimization. The implementation uses
    gradient-based algorithms and embeds a stochastic gradient method
    for global search.
-   [ManifoldOptim](https://cran.r-project.org/package=ManifoldOptim) is an R
    interface to the 'ROPTLIB' optimization library. It optimizes
    real-valued functions over manifolds such as Stiefel, Grassmann, and
    Symmetric Positive Definite matrices.
-   Package [gsl](https://cran.r-project.org/package=gsl) provides BFGS, conjugate
    gradient, steepest descent, and Nelder-Mead algorithms. It uses a
    "line search" approach via the function `multimin()`. It is based on
    the GNU Scientific Library (GSL).
-   An R port of the Scilab neldermead module is packaged in
    [neldermead](https://cran.r-project.org/package=neldermead) offering several
    direct search algorithms based on the simplex approach. And
    [n1qn1](https://cran.r-project.org/package=n1qn1) provides an R port of the
    `n1qn1` optimization procedure in Scilab, a quasi-Newton BFGS method
    without constraints.
-   [optimsimplex](https://cran.r-project.org/package=optimsimplex) provides
    building blocks for simplex-based optimization algorithms such as
    the Nelder-Mead, Spendley, Box method, or multi-dimensional search
    by Torczon, etc.
-   Several derivative-free optimization algorithms are provided with
    package [minqa](https://cran.r-project.org/package=minqa); e.g., the functions
    `bobyqa()`, `newuoa()`, and `uobyqa()` allow to minimize a function
    of many variables by a trust region method that forms quadratic
    models by interpolation. `bobyqa()` additionally permits box
    constraints (bounds) on the parameters.
-   [subplex](https://cran.r-project.org/package=subplex) provides unconstrained
    function optimization based on a subspace searching simplex method.
-   In package [trust](https://cran.r-project.org/package=trust), a routine with the
    same name offers local optimization based on the "trust region"
    approach.
-   [trustOptim](https://cran.r-project.org/package=trustOptim) implements a "trust
    region" algorithm for unconstrained nonlinear optimization. The
    algorithm is optimized for objective functions with sparse Hessians.
    This makes the algorithm highly scalable and efficient, in terms of
    both time and memory footprint.
-   Package [quantreg](https://cran.r-project.org/package=quantreg) contains
    variations of simplex and of interior point routines ( `nlrq()`,
    `crq()`). It provides an interface to L1 regression in the R code of
    function `rq()`.

### <span id="quadratic-optimization">Quadratic Optimization</span>

-   In package [quadprog](https://cran.r-project.org/package=quadprog) `solve.QP()`
    solves quadratic programming problems with linear equality and
    inequality constraints. (The matrix has to be positive definite.)
    [quadprogXT](https://cran.r-project.org/package=quadprogXT) extends this with
    absolute value constraints and absolute values in the objective
    function.
-   [osqp](https://cran.r-project.org/package=osqp) provides bindings to
    [OSQP](https://osqp.org) , the 'Operator Splitting QP' solver from
    the University of Oxford Control Group; it solves sparse convex
    quadratic programming problems with optional equality and inequality
    constraints efficiently.
-   [kernlab](https://cran.r-project.org/package=kernlab) contains the function
    `ipop` for solving quadratic programming problems using interior
    point methods. (The matrix can be positive semidefinite.)
-   [Dykstra](https://cran.r-project.org/package=Dykstra) solves quadratic
    programming problems using R. L. Dykstra's cyclic projection
    algorithm for positive definite and semidefinite matrices. The
    routine allows for a combination of equality and inequality
    constraints.
-   [coneproj](https://cran.r-project.org/package=coneproj) contains routines for
    cone projection and quadratic programming, estimation and inference
    for constrained parametric regression, and shape-restricted
    regression problems.
-   [LowRankQP](https://cran.r-project.org/package=LowRankQP) primal/dual interior
    point method solving quadratic programming problems (especially for
    semidefinite quadratic forms).
-   The COIN-OR project 'qpOASES' implements a reliable QP solver, even
    when tackling semi-definite or degenerated QP problems; it is
    particularly suited for model predictive control (MPC) applications;
    the ROI plugin
    [ROI.plugin.qpoases](https://cran.r-project.org/package=ROI.plugin.qpoases)
    makes it accessible for R users.
-   [limSolve](https://cran.r-project.org/package=limSolve) offers to solve linear
    or quadratic optimization functions, subject to equality and/or
    inequality constraints.

### <span id="optimization-test-functions">Optimization Test Functions</span>

-   Objective functions for benchmarking the performance of global
    optimization algorithms can be found in
    [globalOptTests](https://cran.r-project.org/package=globalOptTests).
-   [smoof](https://cran.r-project.org/package=smoof) has generators for a number of
    both single- and multi-objective test functions that are frequently
    used for benchmarking optimization algorithms; offers a set of
    convenient functions to generate, plot, and work with objective
    functions.
-   [flacco](https://cran.r-project.org/package=flacco) contains tools and features
    used for an Exploratory Landscape Analysis (ELA) of continuous
    optimization problems, capable of quantifying rather complex
    properties, such as the global structure, separability, etc., of the
    optimization problems.
-   [cec2013](https://cran.r-project.org/package=cec2013) and
    [cec2005benchmark](https://cran.r-project.org/package=cec2005benchmark) contain
    many test functions for global optimization from the 2005 and 2013
    special sessions on real-parameter optimization at the IEEE CEC
    congresses on evolutionary computation.
-   Package [<span
    class="GitHub">funconstrain</span>](https://github.com/jlmelville/funconstrain/)
    (on Github) implements 35 of the test functions by More, Garbow, and
    Hillstom, useful for testing unconstrained optimization methods.

### <span id="least-squares-problems">Least-Squares Problems</span>

Function `solve.qr()` (resp. `qr.solve()`) handles over- and
under-determined systems of linear equations, returning least-squares
solutions if possible. And package stats provides `nls()` to determine
least-squares estimates of the parameters of a nonlinear model.
[nls2](https://cran.r-project.org/package=nls2) enhances function `nls()` with brute
force or grid-based searches, to avoid being dependent on starting
parameters or getting stuck in local solutions.

-   Package [nlsr](https://cran.r-project.org/package=nlsr) provides tools for
    working with nonlinear least-squares problems. Functions `nlfb` and
    `nlxb` are intended to eventually supersede the 'nls()' function in
    Base R, by applying a variant of the Marquardt procedure for
    nonlinear least-squares, with bounds constraints and optionally
    Jacobian described as R functions. (It is based on the
    now-deprecated package [nlmrt](https://cran.r-project.org/package=nlmrt).)
-   Package [minpack.lm](https://cran.r-project.org/package=minpack.lm) provides a
    function `nls.lm()` for solving nonlinear least-squares problems by
    a modification of the Levenberg-Marquardt algorithm, with support
    for lower and upper parameter bounds, as found in MINPACK.
-   Package [lsei](https://cran.r-project.org/package=lsei) contains functions that
    solve least-squares linear regression problems under linear
    equality/inequality constraints. Functions for solving quadratic
    programming problems are also available, which transform such
    problems into least squares ones first. (Based on Fortran programs
    of Lawson and Hanson.)
-   Package [nnls](https://cran.r-project.org/package=nnls) interfaces the
    Lawson-Hanson implementation of an algorithm for non-negative
    least-squares, allowing the combination of non-negative and
    non-positive constraints.
-   Package [bvls](https://cran.r-project.org/package=bvls) interfaces the
    Stark-Parker implementation of an algorithm for least-squares with
    upper and lower bounded variables.
-   Package [onls](https://cran.r-project.org/package=onls) implements orthogonal
    nonlinear least-squares regression (ONLS, a.k.a. Orthogonal Distance
    Regression, ODR) using a Levenberg-Marquardt-type minimization
    algorithm based on the ODRPACK Fortran library.
-   [colf](https://cran.r-project.org/package=colf) performs least squares
    constrained optimization on a linear objective function. It contains
    a number of algorithms to choose from and offers a formula syntax
    similar to `lm()`.

### <span id="semidefinite-and-convex-solvers">Semidefinite and Convex Solvers</span>

-   Package [ECOSolveR](https://cran.r-project.org/package=ECOSolveR) provides an
    interface to the Embedded COnic Solver (ECOS), a well-known,
    efficient, and robust C library for convex problems. Conic and
    equality constraints can be specified in addition to integer and
    boolean variable constraints for mixed-integer problems.
-   Package [scs](https://cran.r-project.org/package=scs) applies operator splitting
    to solve linear programs, cone programs (SOCP), and semidefinite
    programs; cones can be second-order, exponential, power cones, or
    any combination of these.
-   [cccp](https://cran.r-project.org/package=cccp) contains routines for solving
    cone constrained convex problems by means of interior-point methods
    (partially ported from Python's CVXOPT).
-   [sdpt3r](https://cran.r-project.org/package=sdpt3r) solves general semidefinite
    Linear Programming (LP) problems, using an R implementation of
    SDPT3, a MATLAB software for semidefinite quadratic-linear programming.
-   The [CLSOCP](https://cran.r-project.org/package=CLSOCP) package provides an
    implementation of a one-step smoothing Newton method for the
    solution of second order cone programming (SOCP) problems.
-   CSDP is a library of routines that implements a primal-dual barrier
    method for solving semidefinite programming problems; it is
    interfaced in the [Rcsdp](https://cran.r-project.org/package=Rcsdp) package.
-   The DSDP library implements an interior-point method for
    semidefinite programming with primal and dual solutions; it is
    interfaced in package [Rdsdp](https://cran.r-project.org/package=Rdsdp).
-   Package [Rmosek](https://cran.r-project.org/package=Rmosek) provides an
    interface to the (commercial) MOSEK optimization library for
    large-scale LP, QP, and MIP problems, with emphasis on (nonlinear)
    conic, semidefinite, and convex tasks; academic licenses are
    available. (An article on Rmosek appeared in the JSS special issue
    on Optimization with R, see below.)

### <span id="global-and-stochastic-optimization">Global and Stochastic Optimization</span>

-   Package [DEoptim](https://cran.r-project.org/package=DEoptim) provides a global
    optimizer based on the Differential Evolution algorithm.
    [RcppDE](https://cran.r-project.org/package=RcppDE) provides a C++
    implementation (using Rcpp) of the same `DEoptim()` function.
-   [DEoptimR](https://cran.r-project.org/package=DEoptimR) provides an
    implementation of the jDE variant of the differential evolution
    stochastic algorithm for nonlinear programming problems (It allows
    to handle constraints in a flexible manner.)
-   The [CEoptim](https://cran.r-project.org/package=CEoptim) package implements a
    cross-entropy optimization technique that can be applied to
    continuous, discrete, mixed, and constrained optimization problems.
-   [GenSA](https://cran.r-project.org/package=GenSA) is a package providing a
    function for generalized Simulated Annealing which can be used to
    search for the global minimum of a quite complex non-linear
    objective function with a large number of optima.
-   [GA](https://cran.r-project.org/package=GA) provides functions for optimization
    using Genetic Algorithms in both, the continuous and discrete case.
    This package allows to run corresponding optimization tasks in
    parallel.
-   Package [genalg](https://cran.r-project.org/package=genalg) contains `rbga()`,
    an implementation of a genetic algorithm for multi-dimensional
    function optimization.
-   Package [rgenoud](https://cran.r-project.org/package=rgenoud) offers `genoud()`,
    a routine which is capable of solving complex function
    minimization/maximization problems by combining evolutionary
    algorithms with a derivative-based (quasi-Newtonian) approach.
-   The [Jaya](https://cran.r-project.org/package=Jaya) package provides an
    implementation of the Jaya algorithm, a population based heuristic algorithm
    which repeatedly modifies a population by looking at best and worst solutions.
-   Machine coded genetic algorithm (MCGA) provided by package
    [mcga](https://cran.r-project.org/package=mcga) is a tool which solves
    optimization problems based on byte representation of variables.
-   A particle swarm optimizer (PSO) is implemented in package
    [pso](https://cran.r-project.org/package=pso), and also in
    [psoptim](https://cran.r-project.org/package=psoptim). Another (parallelized)
    implementation of the PSO algorithm can be found in package `ppso`
    available from [rforge.net/ppso](https://www.rforge.net/ppso/) .
-   Package [hydroPSO](https://cran.r-project.org/package=hydroPSO) implements the
    latest Standard Particle Swarm Optimization algorithm (SPSO-2011);
    it is parallel-capable, and includes several fine-tuning options and
    post-processing functions.
-   [<span class="GitHub">hydromad</span>](https://github.com/floybix/hydromad/)
    (on Github) contains the `SCEoptim` function for Shuffled Compex
    Evolution (SCE) optimization, an evolutionary algorithm, combined
    with a simplex method.
-   Package [ABCoptim](https://cran.r-project.org/package=ABCoptim) implements the
    Artificial Bee Colony (ABC) optimization approach.
-   Package [metaheuristicOpt](https://cran.r-project.org/package=metaheuristicOpt)
    contains implementations of several evolutionary optimization
    algorithms, such as particle swarm, dragonfly and firefly, sine
    cosine algorithms and many others.
-   Package [ecr](https://cran.r-project.org/package=ecr) provides a framework for
    building evolutionary algorithms for single- and multi-objective
    continuous or discrete optimization problems.
-   CMA-ES by N. Hansen, global optimization procedure using a
    covariance matrix adapting evolutionary strategy, is implemented in
    several packages: In packages [cmaes](https://cran.r-project.org/package=cmaes)
    and [cmaesr](https://cran.r-project.org/package=cmaesr), in
    [parma](https://cran.r-project.org/package=parma) as `cmaes`, in
    [adagio](https://cran.r-project.org/package=adagio) as `pureCMAES`, and in
    [rCMA](https://cran.r-project.org/package=rCMA) as `cmaOptimDP`, interfacing
    Hansen's own Java implementation.
-   Package [Rmalschains](https://cran.r-project.org/package=Rmalschains) implements
    an algorithm family for continuous optimization called memetic
    algorithms with local search chains (MA-LS-Chains).
-   An R implementation of the Self-Organising Migrating Algorithm
    (SOMA) is available in package [soma](https://cran.r-project.org/package=soma).
    This stochastic optimization method is somewhat similar to genetic algorithms.
-   [nloptr](https://cran.r-project.org/package=nloptr) supports several global
    optimization routines, such as DIRECT, controlled random search
    (CRS), multi-level single-linkage (MLSL), improved stochastic
    ranking (ISR-ES), or stochastic global optimization (StoGO).
-   The [NMOF](https://cran.r-project.org/package=NMOF) package provides
    implementations of differential evolution, particle swarm
    optimization, local search and threshold accepting (a variant of
    simulated annealing). The latter two methods also work for discrete
    optimization problems, as does the implementation of a genetic
    algorithm that is included in the package.
-   [OOR](https://cran.r-project.org/package=OOR) implements optimistic
    optimization methods for global optimization of deterministic or stochastic
    functions (in small dimensions).
-   [SACOBRA](https://cran.r-project.org/package=SACOBRA) is a package for numeric
    constrained optimization of expensive black-box functions under
    severely limited budgets; it implements an extension of the COBRA
    algorithm with initial design generation and self-adjusting random restarts.
-   [RCEIM](https://cran.r-project.org/package=RCEIM) implements a stochastic
    heuristic method for performing multi-dimensional function optimization.

<span id="mathematical-programming-solvers">Mathematical Programming Solvers</span>
-----------------------------------------------------------------------------------

This section provides an overview of open source as well as commercial
optimizers. Which type of mathematical programming problem can be solved
by a certain package or function can be seen from the abbreviations in
square brackets. For a [Classification According to
Subject](#classification-according-to-subject) see the list at the end
of this task view.

-   Package [ompr](https://cran.r-project.org/package=ompr) is an optimization
    modeling package to model and solve Mixed Integer Linear Programs in
    an algebraic way directly in R. The models are solver-independent
    and thus offer the possibility to solve models with different
    solvers. (Inspired by Julia's JuMP project.)
-   [linprog](https://cran.r-project.org/package=linprog) solves linear programming
    problems using the function `solveLP()` (the solver is based on
    [lpSolve](https://cran.r-project.org/package=lpSolve)) and can read model files
    in MPS format.
-   In the [boot](https://cran.r-project.org/package=boot) package there is a
    routine called `simplex()` which realizes the two-phase tableau
    simplex method for (relatively small) linear programming problems.
-   [rcdd](https://cran.r-project.org/package=rcdd) offers the function `lpcdd()`
    for solving linear programs with exact arithmetic using the [GNU
    Multiple Precision (GMP)](https://gmplib.org) library.

<!-- -->

-   The [NEOS Server for
    Optimization](https://www.neos-server.org/neos/) provides online
    access to state-of-the-art optimization problem solvers. Package
    [rneos](https://cran.r-project.org/package=rneos) enables the user to pass
    optimization problems to NEOS and retrieve results within R.

### <span id="interfaces-to-open-source-optimizers">Interfaces to Open Source Optimizers</span>

-   Package [clpAPI](https://cran.r-project.org/package=clpAPI) provides high level
    access from R to low-level API routines of the [COIN OR
    Clp](https://projects.coin-or.org/Clp) solver library.
-   Package [lpSolve](https://cran.r-project.org/package=lpSolve) contains the
    routine `lp()` to solve LPs and MILPs by calling the freely
    available solver [lp\_solve](http://lpsolve.sourceforge.net) . This
    solver is based on the revised simplex method and a branch-and-bound
    (B&B) approach. It supports semi-continuous variables and Special
    Ordered Sets (SOS). Furthermore `lp.assign()` and `lp.transport()`
    are aimed at solving assignment problems and transportation
    problems, respectively. Additionally, there is the package
    [lpSolveAPI](https://cran.r-project.org/package=lpSolveAPI) which provides an R
    interface to the low level API routines of lp\_solve (see also
    project [<span
    class="Rforge">lpsolve</span>](https://R-Forge.R-project.org/projects/lpsolve/)
    on R-Forge). [lpSolveAPI](https://cran.r-project.org/package=lpSolveAPI)
    supports reading linear programs from files in lp and MPS format.
-   Packages [glpkAPI](https://cran.r-project.org/package=glpkAPI) as well as
    package [Rglpk](https://cran.r-project.org/package=Rglpk) provide an interface
    to the [GNU Linear Programming
    Kit](https://www.gnu.org/software/glpk/) (GLPK). Whereas the former
    provides high level access to low level routines the latter offers a
    high level routine `Rglpk_solve_LP()` to solve MILPs using GLPK.
    Both packages offer the possibility to use models formulated in the
    MPS format.
-   [Rsymphony](https://cran.r-project.org/package=Rsymphony) has the routine
    `Rsymphony_solve_LP()` that interfaces the SYMPHONY solver for
    mixed-integer linear programs. (SYMPHONY is part of the
    [Computational Infrastructure for Operations
    Research](http://www.coin-or.org/) (COIN-OR) project.) Package
    `lpsymphony` in Bioconductor provides a similar interface to SYMPHONY
    that is easier to install.
-   The NOMAD solver is implemented in the
    [crs](https://cran.r-project.org/package=crs) package for solving mixed integer
    programming problems. This algorithm is accessible via the
    `snomadr()` function and is primarily designed for constrained
    optimization of blackbox functions.
-   'Clp' and 'Cbc' are open source solvers from the COIN-OR suite.
    'Clp' solves linear programs with continuous objective variables and
    is available through
    [ROI.plugin.clp](https://cran.r-project.org/package=ROI.plugin.clp). 'Cbc' is a
    powerful mixed integer linear programming solver (based on 'Clp');
    package 'rcbc' can be installed from: [<span
    class="GitHub">rcbc</span>](https://github.com/dirkschumacher/rcbc/)
    (on Github).

### <span id="interfaces-to-commercial-optimizers">Interfaces to Commercial Optimizers</span>

This section surveys interfaces to commercial solvers. Typically, the
corresponding libraries have to be installed separately.

-   Packages [cplexAPI](https://cran.r-project.org/package=cplexAPI) and
    [Rcplex](https://cran.r-project.org/package=Rcplex) provide interfaces to the
    IBM [CPLEX
    Optimizer](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/)
    . CPLEX provides dual/primal simplex optimizers as well as a barrier
    optimizer for solving large scale linear and quadratic programs. It
    offers a mixed integer optimizer to solve difficult mixed integer
    programs including (possibly non-convex) MIQCP. Note that CPLEX is
    **not free** and you have to get a license. Academics will receive a
    free licence upon request.
-   The API of the commercial solver LINDO can be accessed in R via
    package [rLindo](https://cran.r-project.org/package=rLindo). The [LINDO
    API](http://www.lindo.com/) allows for solving linear, integer,
    quadratic, conic, general nonlinear, global and stochastic
    programming problems.
-   Package [Rmosek](https://cran.r-project.org/package=Rmosek) offers an interface
    to the commercial optimizer from [MOSEK](https://www.mosek.com/) .
    It provides dual/primal simplex optimizers as well as a barrier
    optimizer. In addition to solving LP and QP problems this solver can
    handle SOCP and quadratically constrained programming (QPQC) tasks.
    Furthermore, it offers a mixed integer optimizer to solve difficult
    mixed integer programs (MILP, MISOCP, etc.). You have to get a
    license, but Academic licenses are free of charge.
-   Gurobi Optimization ships an R binding since their 5.0 release that
    allows to solve LP, MIP, QP, MIQP, SOCP, and MISOCP models from
    within R. See the [R with
    Gurobi](https://www.gurobi.com/products/modeling-languages/r)
    website for more details.
-   The [localsolver](https://cran.r-project.org/package=localsolver) package
    provides an interface to the hybrid mathematical programming
    software [LocalSolver](http://www.localsolver.com/) from Innovation
    24. LocalSolver is a commercial product, academic licenses are
    available on request.

<span id="combinatorial-optimization">Combinatorial Optimization</span>
-----------------------------------------------------------------------

-   Package [adagio](https://cran.r-project.org/package=adagio) provides R functions
    for single and multiple knapsack problems, and solves subset sum and
    assignment tasks.
-   In package [clue](https://cran.r-project.org/package=clue) `solve_LSAP()`
    enables the user to solve the linear sum assignment problem (LSAP)
    using an efficient C implementation of the Hungarian algorithm.
-   [FLSSS](https://cran.r-project.org/package=FLSSS) provides multi-threaded
    solvers for fixed-size single and multi dimensional subset sum
    problems with optional constraints on target sum and element range,
    fixed-size single and multi dimensional knapsack problems, binary
    knapsack problems and generalized assignment problems via exact
    algorithms or metaheuristics.
-   Package [qap](https://cran.r-project.org/package=qap) solves Quadratic
    Assignment Problems (QAP) applying a simulated annealing heuristics
    (other approaches will follow).
-   [igraph](https://cran.r-project.org/package=igraph), a package for graph and
    network analysis, uses the very fast igraph C library. It can be
    used to calculate shortest paths, maximal network flows, minimum
    spanning trees, etc.
-   [mknapsack](https://cran.r-project.org/package=mknapsack) solves multiple
    knapsack problems, based on LP solvers such as 'lpSolve' or 'CBC';
    will assign items to knapsacks in a way that the value of the top
    knapsacks is as large as possible.
-   Package 'knapsack' (see R-Forge project [<span
    class="Rforge">optimist</span>](https://R-Forge.R-project.org/projects/optimist/))
    provides routines from the book \`Knapsack Problems' by Martello and
    Toth. There are functions for (multiple) knapsack, subset sum and
    binpacking problems. (Use of Fortran codes is restricted to personal
    research and academic purposes only.)
-   [nilde](https://cran.r-project.org/package=nilde) provides routines for
    enumerating all integer solutions of linear Diophantine equations,
    resp. all solutions of knapsack, subset sum, and additive
    partitioning problems (based on a generating functions approach).
-   [matchingR](https://cran.r-project.org/package=matchingR) and
    [matchingMarkets](https://cran.r-project.org/package=matchingMarkets) implement
    the Gale-Shapley algorithm for the stable marriage and the college
    admissions problem, the stable roommates and the house allocation
    problem.
-   Package [optmatch](https://cran.r-project.org/package=optmatch) provides
    routines for solving matching problems by translating them into
    minimum-cost flow problems and then solved optimaly by the RELAX-IV
    codes of Bertsekas and Tseng (free for research).
-   Package [TSP](https://cran.r-project.org/package=TSP) provides basic
    infrastructure for handling and solving the traveling salesperson
    problem (TSP). The main routine `solve_TSP()` solves the TSP through
    several heuristics. In addition, it provides an interface to the
    [Concorde TSP Solver](http://www.tsp.gatech.edu/concorde/index.html),
    which has to be downloaded separately.

<span id="multi-objective-optimization">Multi Objective Optimization</span>
---------------------------------------------------------------------------

-   Function `caRamel` in package
    [caRamel](https://cran.r-project.org/package=caRamel) is a multi-objective
    optimizer, applying a combination of the multiobjective evolutionary
    annealing-simplex (MEAS) method and the non-dominated sorting
    genetic algorithm (NGSA-II); it was initially developed for the
    calibration of hydrological models.
-   Multi-criteria optimization problems can be solved using package
    [mco](https://cran.r-project.org/package=mco) which implements genetic algorithms.
-   [GPareto](https://cran.r-project.org/package=GPareto)
    provides multi-objective optimization algorithms for expensive black-box
    functions and uncertainty quantification methods.

<span id="specific-applications-in-optimization">Specific Applications in Optimization</span>
---------------------------------------------------------------------------------------------

-   The data cloning algorithm is a global optimization approach and a
    variant of simulated annealing which has been implemented in package
    [dclone](https://cran.r-project.org/package=dclone). The package provides low
    level functions for implementing maximum likelihood estimating
    procedures for complex models using data cloning and Bayesian Markov
    chain Monte Carlo methods.
-   [irace](https://cran.r-project.org/package=irace) contains an optimization
    algorithm for optimizing the parameters of other optimization
    algorithms. This problem is called "(offline) algorithm
    configuration".
-   Package [kofnGA](https://cran.r-project.org/package=kofnGA) uses a genetic
    algorithm to choose a subset of a fixed size k from the integers
    1:n, such that a user- supplied objective function is minimized at
    that subset.
-   [copulaedas](https://cran.r-project.org/package=copulaedas) provides a platform
    where 'estimation of distribution algorithms' (EDA) based on copulas
    can be implemented and studied; the package offers various EDAs, and
    newly developed EDAs can be integrated by extending an S4 class.
-   [tabuSearch](https://cran.r-project.org/package=tabuSearch) implements a tabu
    search algorithm for optimizing binary strings, maximizing a user
    defined target function, and returns the best (i.e. maximizing)
    binary configuration found.
-   Besides functionality for solving general isotone regression
    problems, package [isotone](https://cran.r-project.org/package=isotone) provides
    a framework of active set methods for isotone optimization problems
    with arbitrary order restrictions.
-   [mlrMBO](https://cran.r-project.org/package=mlrMBO) is a flexible and
    comprehensive R toolbox for model-based optimization ('MBO'), also
    known as Bayesian optimization. And
    [rBayesianOptimization](https://cran.r-project.org/package=rBayesianOptimization)
    is an implementation of Bayesian global optimization with Gaussian
    Processes, for parameter tuning and optimization of hyperparameters.
-   The [desirability](https://cran.r-project.org/package=desirability) package
    contains S3 classes for multivariate optimization using the
    desirability function approach of Harrington (1965) using functional
    forms described by Derringer and Suich (1980).
-   Package [sna](https://cran.r-project.org/package=sna) contains the function
    `lab.optim()` which is the front-end to a series of heuristic
    routines for optimizing some bivariate graph statistic.
-   [maxLik](https://cran.r-project.org/package=maxLik) adds a likelihood-specific
    layer on top of a number of maximization routines like
    Brendt-Hall-Hall-Hausman (BHHH) and Newton-Raphson among others. It
    includes summary and print methods which extract the standard errors
    based on the Hessian matrix and allows easy swapping of maximization
    algorithms. It also provides a function to check whether an analytic
    derivative is computed directly.


<span id="related-links">Related Links</span>
---------------------------------------------

-   [Journal of Statistical Software Special Volume on Optimization
    (Editor: Ravi Varadhan)](https://www.jstatsoft.org/v60)
-   [Nonlinear Parameter Optimization Using R Tools -- John C. Nash
    (Wiley)](http://www.wiley.com/WileyCDA/WileyTitle/productCd-1118569288.html)
-   [Modern Optimization With R -- Paulo Cortez (Springer UseR
    Series)](https://www.springer.com/mathematics/book/978-3-319-08262-2)
-   [Yet Another Math Programming
    Consultant](http://yetanothermathprogrammingconsultant.blogspot.com)
-   [COIN-OR Project](http://www.coin-or.org/)
-   [NEOS Optimization
    Guide](http://www.neos-guide.org/Optimization-Guide)
-   [Decision Tree for Optimization
    Software](http://plato.asu.edu/sub/pns.html)
-   [Mathematics Subject Classification - Mathematical
    programming](http://www.ams.org/mathscinet/msc/msc2010.html?t=90Cxx&btn=Current)
