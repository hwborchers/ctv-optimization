CRAN Task View: Optimization and Mathematical Programming
---------------------------------------------------------

|                 |                                                |
|-----------------|------------------------------------------------|
| **Maintainer:** | Stefan Theussl and Hans W. Borchers            |
| **Contact:**    | R-optimization at mailbox.org                  |
| **Version:**    | 2018-04-26                                     |
| **URL:**        | <https://CRAN.R-project.org/view=Optimization> |


This CRAN task view contains a list of packages which offer facilities for solving optimization problems. Although every regression model in statistics solves an optimization problem they are not part of this view. If you are looking for regression methods, the following views will contain useful starting points: [Multivariate](https://cran.r-project.org/web/views/Multivariate.html), [SocialSciences](https://cran.r-project.org/web/views/SocialSciences.html), [Robust](https://cran.r-project.org/web/views/Robust.html) among others. The focus of this task view is on [Optimization Infrastructure Packages](#optimization-infrastructure-packages), [General Purpose Continuous Solvers](#general-purpose-continuous-solvers), [Mathematical Programming Solvers](#mathematical-programming-solvers), and [Specific Applications in Optimization](#specific-applications-in-optimization). Packages are categorized in these four sections.

Many packages provide functionality for more than one of the subjects listed at the end of this task view. E.g., mixed integer linear programming solvers typically offer standard linear programming routines like the simplex algorithm. Therefore following each package description a list of abbreviations describes the typical features of the optimizer (i.e., the problems which can be solved). The full names of the abbreviations given in square brackets can be found at the end of this task view under [Classification According to Subject](#classification-according-to-subject).

If you think that some package is missing from the list, please let us know.

<span id="optimization-infrastructure-packages">Optimization Infrastructure Packages</span>
-------------------------------------------------------------------------------------------

-   Trying to unify optimization algorithms via a single wrapper function, [optimr](https://cran.r-project.org/package=optimr/index.html) (and 'optimrx' on R-Forge) helps to proper specify (nonlinear) optimization problems, including objective function, gradient function, and scaling. It supports the (local) optimization of smooth, nonlinear functions with at most box/bound constraints. Function `opm`, returning a dataframe, compares solvers for a selected optimization task. (Note that [optimx](../packages/optimx) will get deprecated.)

-   The R Optimization Infrastructure ([ROI](https://cran.r-project.org/package=ROI)) package provides a framework for handling optimization problems in R. It uses an object-oriented approach to define and solve various optimization tasks in R which can be from different problem classes (e.g., linear, quadratic, non-linear programming problems). This makes optimization transparent for the R user as the corresponding workflow is completely abstracted from the underlying solver. The approach allows for easy switching between solvers and thus enhances comparability.

-   The package [CVXR](https://cran.r-project.org/package=CVXR) provides an object-oriented modeling language for Disciplined Convex Programming (DCP). It allows the user to formulate convex optimization problems in a natural way following mathematical convention and DCP rules. The system analyzes the problem, verifies its convexity, converts it into a canonical form, and hands it off to an appropriate solver such as ECOS or SCS to obtain the solution. (CVXR is derived from the well-known MATLAB toolbox CVX, developed at Stanford University.)

<span id="general-purpose-continuous-solvers">General Purpose Continuous Solvers</span>
---------------------------------------------------------------------------------------

Package stats offers several general purpose optimization routines. For one-dimensional unconstrained function optimization there is `optimize()` which searches an interval for a minimum or maximum. Function `optim()` provides an implementation of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) method, bounded BFGS, conjugate gradient (CG), Nelder-Mead, and simulated annealing (SANN) optimization methods. It utilizes gradients, if provided, for faster convergence. Typically it is used for unconstrained optimization but includes an option for box-constrained optimization.

Additionally, for minimizing a function subject to linear inequality constraints, stats contains the routine `constrOptim()`. Then there is `nlm` which is used for solving nonlinear unconstrained minimization problems. `nlminb()` offers box-constrained optimization using the PORT routines. \[RGA, QN\]

-   [lbfgsb3](https://cran.r-project.org/package=lbfgsb3) interfaces the J.Nocedal et al. L-BFGS-B 3.0 Fortran code, a limited memory BFGS solver, allowing bound constraints and being applicable to higher-dimensional problems. \[QN\]
-   And [lbfgs](https://cran.r-project.org/package=lbfgs) wraps the libBFGS library by N. Okazaki (converted from Nocedal's library), interfacing both the L-BFGS and the OWL-QN algorithm, the latter being particularly suited for higher-dimensional problems. \[QN\]
-   [RcppNumerical](https://cran.r-project.org/package=RcppNumerical) is a collection of open source libraries for numerical computing and their integration with 'Rcpp'. It provides a wrapper for the L-BFGS algorithm, based on the LBFGS++ library (based on code of N. Okazaki).
-   The following packages implement optimization routines in pure R, for nonlinear functions with bounds constraints. [Rcgmin](https://cran.r-project.org/package=Rcgmin/index.html): gradient function minimization similar to GC; [Rvmmin](../packages/Rvmmin/index.html): variable metric function minimization; [Rtnmin](../packages/Rtnmin): truncated Newton function minimization.
-   Package [ucminf](https://cran.r-project.org/package=ucminf) implements an algorithm of quasi-Newton type for nonlinear unconstrained optimization, combining a trust region with line search approaches. The interface of `ucminf()` is designed for easy interchange with `optim()`.\[QN\]
-   [mize](https://cran.r-project.org/package=mize) implements optimization algorithms in pure R, including conjugate gradient (CG), Broyden-Fletcher-Goldfarb-Shanno (BFGS) and limited memory BFGS (L-BFGS) methods. Most internal parameters can be set through the calling interface.
-   Package [nloptr](https://cran.r-project.org/package=nloptr) provides access to NLopt, an LGPL licensed library of various nonlinear optimization algorithms. It includes local derivative-free (COBYLA, Nelder-Mead, Subplex) and gradient-based (e.g., BFGS) methods, and also the augmented Lagrangian approach for nonlinear constraints. \[DF, GO, QN\]
-   Package [dfoptim](https://cran.r-project.org/package=dfoptim), derivative-free optimization procedures, contains quite efficient R implementations of the Nelder-Mead and Hooke-Jeeves algorithms (unconstrained and bounds-constrained). \[DF\]
-   Implementations of the augmented Lagrange barrier minimization algorithm for optimizing smooth nonlinear objective functions (with equality and inequality constraints) can be found in packages [alabama](https://cran.r-project.org/package=alabama/index.html) and [Rsolnp](../packages/Rsolnp).
-   [NlcOptim](https://cran.r-project.org/package=NlcOptim) solves nonlinear optimization problems with linear and nonlinear equality and inequality constraints, implementing a Sequential Quadratic Programming (SQP) method.
-   In package Rdonlp2 (see the [<span class="Rforge">rmetrics</span>](https://R-Forge.R-project.org/projects/rmetrics/) project) function `donlp2()`, a wrapper for the DONLP2 solver, offers the minimization of smooth nonlinear functions and constraints. DONLP2 can be used freely for any kind of research purposes, otherwise it requires licensing. \[GO, NLP\]
-   [clue](https://cran.r-project.org/package=clue) contains the function `sumt()` for solving constrained optimization problems via the sequential unconstrained minimization technique (SUMT).
-   [BB](https://cran.r-project.org/package=BB/index.html) contains the function `spg()` providing a spectral projected gradient method for large scale optimization with simple constraints. It takes a nonlinear objective function as an argument as well as basic constraints. Furthermore, [BB](../packages/BB) contains two functions ( `dfsane()` and `sane()`) for using the spectral gradient method for solving a nonlinear system of equations.
-   [GrassmannOptim](https://cran.r-project.org/package=GrassmannOptim) is a package for Grassmann manifold optimization. The implementation uses gradient-based algorithms and embeds a stochastic gradient method for global search.
-   [ManifoldOptim](https://cran.r-project.org/package=ManifoldOptim) is an R interface to the 'ROPTLIB' optimization library. It optimizes real-valued functions over manifolds such as Stiefel, Grassmann, and Symmetric Positive Definite matrices.
-   Package [gsl](https://cran.r-project.org/package=gsl) provides BFGS, conjugate gradient, steepest descent, and Nelder-Mead algorithms. It uses a "line search" approach via the function `multimin()`. It is based on the GNU Scientific Library (GSL). \[RGA, QN\]
-   An R port of the Scilab neldermead module is packaged in [neldermead](https://cran.r-project.org/package=neldermead/index.html) offering several direct search algorithms based on the simplex approach. And [n1qn1](../packages/n1qn1) provides an R port of the `n1qn1` optimization procedure in Scilab, a quasi-Newton BFGS method without constraints.
-   [optimsimplex](https://cran.r-project.org/package=optimsimplex) provides building blocks for simplex-based optimization algorithms such as the Nelder-Mead, Spendley, Box method, or multi-dimensional search by Torczon, etc.
-   Several derivative-free optimization algorithms are provided with package [minqa](https://cran.r-project.org/package=minqa); e.g., the functions `bobyqa()`, `newuoa()`, and `uobyqa()` allow to minimize a function of many variables by a trust region method that forms quadratic models by interpolation. `bobyqa()` additionally permits box constraints (bounds) on the parameters. \[DF\]
-   Package [powell](https://cran.r-project.org/package=powell) optimizes functions using Powell's UObyQA algorithm (Unconstrained Optimization by Quadratic Approximation).
-   [subplex](https://cran.r-project.org/package=subplex) provides unconstrained function optimization based on a subspace searching simplex method.
-   In package [trust](https://cran.r-project.org/package=trust), a routine with the same name offers local optimization based on the "trust region" approach.
-   [trustOptim](https://cran.r-project.org/package=trustOptim) implements a "trust region" algorithm for unconstrained nonlinear optimization. The algorithm is optimized for objective functions with sparse Hessians. This makes the algorithm highly scalable and efficient, in terms of both time and memory footprint.
-   Package [quantreg](https://cran.r-project.org/package=quantreg) contains variations of simplex and of interior point routines ( `nlrq()`, `crq()`). It provides an interface to L1 regression in the R code of function `rq()`. \[SPLP, LP, IPM\]

### <span id="quadratic-optimization">Quadratic Optimization</span>

-   In package [quadprog](https://cran.r-project.org/package=quadprog/index.html) `solve.QP()` solves quadratic programming problems with linear equality and inequality constraints. (The matrix has to be positive definite.) [quadprogXT](../packages/quadprogXT) extends this with absolute value constraints and absolute values in the objective function. \[QP\]
-   [kernlab](https://cran.r-project.org/package=kernlab) contains the function `ipop` for solving quadratic programming problems using interior point methods. (The matrix can be positive semidefinite.) \[IPM, QP\]
-   [Dykstra](https://cran.r-project.org/package=Dykstra) solves quadratic programming problems using R. L. Dykstra's cyclic projection algorithm for positive definite and semidefinite matrices. The routine allows for a combination of equality and inequality constraints. \[QP\]
-   [rosqp](https://cran.r-project.org/package=rosqp) provides bindings to the 'OSQP' solver, the 'Operator Splitting QP Solver' of the University of Oxford Control Group, which can solve sparse convex quadratic programming problems with optional equality and inequality constraints. \[QP\]
-   [coneproj](https://cran.r-project.org/package=coneproj) contains routines for cone projection and quadratic programming, estimation and inference for constrained parametric regression, and shape-restricted regression problems. \[QP\]
-   [LowRankQP](https://cran.r-project.org/package=LowRankQP) primal/dual interior point method solving quadratic programming problems (especially for semidefinite quadratic forms). \[IPM, QP\]
-   The COIN-OR project \[qpOASES\](https://projects.coin-or.org/qpOASES/) implements a reliable QP solver, even when tackling semi-definite or degenerated QP problems; it is particularly suited for model predictive control (MPC) applications; the ROI plugin [ROI.plugin.qpoases](https://cran.r-project.org/package=ROI.plugin.qpoases) makes it accessible for R users. \[QP\]
-   [limSolve](https://cran.r-project.org/package=limSolve) offers to solve linear or quadratic optimization functions, subject to equality and/or inequality constraints. \[LP, QP\]

### <span id="optimization-test-functions">Optimization Test Functions</span>

-   Objective functions for benchmarking the performance of global optimization algorithms can be found in [globalOptTests](https://cran.r-project.org/package=globalOptTests).
-   [smoof](https://cran.r-project.org/package=smoof) has generators for a number of both single- and multi-objective test functions that are frequently used for benchmarking optimization algorithms; offers a set of convenient functions to generate, plot, and work with objective functions.
-   [flacco](https://cran.r-project.org/package=flacco) contains tools and features used for an Exploratory Landscape Analysis (ELA) of continuous optimization problems, capable of quantifying rather complex properties, such as the global structure, separability, etc., of the optimization problems.
-   [cec2013](https://cran.r-project.org/package=cec2013/index.html) and [cec2005benchmark](../packages/cec2005benchmark) contain many test functions for global optimization from the 2005 and 2013 special sessions on real-parameter optimization at the IEEE CEC congresses on evolutionary computation.
-   Package [<span class="GitHub">funconstrain</span>](https://github.com/jlmelville/funconstrain/) (on Github) implements 35 of the test functions by More, Garbow, and Hillstom, useful for testing unconstrained optimization methods.

### <span id="least-squares-problems">Least-Squares Problems</span>

Function `solve.qr()` (resp. `qr.solve()`) handles over- and under-determined systems of linear equations, returning least-squares solutions if possible. And package stats provides `nls()` to determine least-squares estimates of the parameters of a nonlinear model. [nls2](https://cran.r-project.org/package=nls2) enhances function `nls()` with brute force or grid-based searches, to avoid being dependent on starting parameters or getting stuck in local solutions.

-   Package [nlsr](https://cran.r-project.org/package=nlsr/index.html) provides tools for working with nonlinear least-squares problems. Functions `nlfb` and `nlxb` are intended to eventually supersede the 'nls()' function in Base R, by applying a variant of the Marquardt procedure for nonlinear least-squares, with bounds constraints and optionally Jacobian described as R functions. (It is based on the now-deprecated package [nlmrt](../packages/nlmrt).)
-   Package [minpack.lm](https://cran.r-project.org/package=minpack.lm) provides a function `nls.lm()` for solving nonlinear least-squares problems by a modification of the Levenberg-Marquardt algorithm, with support for lower and upper parameter bounds, as found in MINPACK.
-   Package [lsei](https://cran.r-project.org/package=lsei) contains functions that solve least-squares linear regression problems under linear equality/inequality constraints. Functions for solving quadratic programming problems are also available, which transform such problems into least squares ones first. (Based on Fortran programs of Lawson and Hanson.)
-   Package [nnls](https://cran.r-project.org/package=nnls) interfaces the Lawson-Hanson implementation of an algorithm for non-negative least-squares, allowing the combination of non-negative and non-positive constraints.
-   Package [bvls](https://cran.r-project.org/package=bvls) interfaces the Stark-Parker implementation of an algorithm for least-squares with upper and lower bounded variables.
-   Package [onls](https://cran.r-project.org/package=onls) implements orthogonal nonlinear least-squares regression (ONLS, a.k.a. Orthogonal Distance Regression, ODR) using a Levenberg-Marquardt-type minimization algorithm based on the ODRPACK Fortran library.
-   [colf](https://cran.r-project.org/package=colf) performs least squares constrained optimization on a linear objective function. It contains a number of algorithms to choose from and offers a formula syntax similar to `lm()`.

### <span id="semidefinite-and-convex-solvers">Semidefinite and Convex Solvers</span>

-   Package [ECOSolveR](https://cran.r-project.org/package=ECOSolveR) provides an interface to the Embedded COnic Solver (ECOS), a well-known, efficient, and robust C library for convex problems. Conic and equality constraints can be specified in addition to integer and boolean variable constraints for mixed-integer problems.
-   Package [scs](https://cran.r-project.org/package=scs) applies operator splitting to solve linear programs, cone programs (SOCP), and semidefinite programs; cones can be second-order, exponential, power cones, or any combination of these.
-   [cccp](https://cran.r-project.org/package=cccp) contains routines for solving cone constrained convex problems by means of interior-point methods (partially ported from Python's CVXOPT).
-   [sdpt3r](https://cran.r-project.org/package=sdpt3r) solves general semidefinite Linear Programming (LP) problems, using an R implementation of SDPT3, a MATLAB software for semidefinite quadratic-linear programming.
-   The [CLSOCP](https://cran.r-project.org/package=CLSOCP) package provides an implementation of a one-step smoothing Newton method for the solution of second order cone programming (SOCP) problems.
-   CSDP is a library of routines that implements a primal-dual barrier method for solving semidefinite programming problems; it is interfaced in the [Rcsdp](https://cran.r-project.org/package=Rcsdp) package. \[SDP\]
-   The DSDP library implements an interior-point method for semidefinite programming with primal and dual solutions; it is interfaced in package [Rdsdp](https://cran.r-project.org/package=Rdsdp). \[SDP\]
-   Package [Rmosek](https://cran.r-project.org/package=Rmosek) provides an interface to the (commercial) MOSEK optimization library for large-scale LP, QP, and MIP problems, with emphasis on (nonlinear) conic, semidefinite, and convex tasks; academic licenses are available. (An article on Rmosek appeared in the JSS special issue on Optimization with R, see below.) \[SDP, CP\]

### <span id="global-and-stochastic-optimization">Global and Stochastic Optimization</span>

-   Package [DEoptim](https://cran.r-project.org/package=DEoptim/index.html) provides a global optimizer based on the Differential Evolution algorithm. [RcppDE](../packages/RcppDE) provides a C++ implementation (using Rcpp) of the same `DEoptim()` function.
-   [DEoptimR](https://cran.r-project.org/package=DEoptimR) provides an implementation of the jDE variant of the differential evolution stochastic algorithm for nonlinear programming problems (It allows to handle constraints in a flexible manner.)
-   The [CEoptim](https://cran.r-project.org/package=CEoptim) package implements a cross-entropy optimization technique that can be applied to continuous, discrete, mixed, and constrained optimization problems. \[COP\]
-   [GenSA](https://cran.r-project.org/package=GenSA) is a package providing a function for generalized Simulated Annealing which can be used to search for the global minimum of a quite complex non-linear objective function with a large number of optima.
-   [GA](https://cran.r-project.org/package=GA) provides functions for optimization using Genetic Algorithms in both, the continuous and discrete case. This package allows to run corresponding optimization tasks in parallel.
-   Package [genalg](https://cran.r-project.org/package=genalg) contains `rbga()`, an implementation of a genetic algorithm for multi-dimensional function optimization.
-   Package [rgenoud](https://cran.r-project.org/package=rgenoud) offers `genoud()`, a routine which is capable of solving complex function minimization/maximization problems by combining evolutionary algorithms with a derivative-based (quasi-Newtonian) approach.
-   Machine coded genetic algorithm (MCGA) provided by package [mcga](https://cran.r-project.org/package=mcga) is a tool which solves optimization problems based on byte representation of variables.
-   A particle swarm optimizer (PSO) is implemented in package [pso](https://cran.r-project.org/package=pso/index.html), and also in [psoptim](../packages/psoptim). Another (parallelized) implementation of the PSO algorithm can be found in package `ppso` available from [rforge.net/ppso](https://www.rforge.net/ppso/).
-   Package [hydroPSO](https://cran.r-project.org/package=hydroPSO) implements the latest Standard Particle Swarm Optimization algorithm (SPSO-2011); it is parallel-capable, and includes several fine-tuning options and post-processing functions.
-   [<span class="GitHub">hydromad</span>](https://github.com/floybix/hydromad/) (on Github) contains the `SCEoptim` function for Shuffled Compex Evolution (SCE) optimization, an evolutionary algorithm, combined with a simplex method.
-   Package [ABCoptim](https://cran.r-project.org/package=ABCoptim) implements the Artificial Bee Colony (ABC) optimization approach.
-   Package [metaheuristicOpt](https://cran.r-project.org/package=metaheuristicOpt) contains implementations of several evolutionary optimization algorithms, such as particle swarm, dragonfly and firefly, sine cosine algorithms and many others.
-   Package [ecr](https://cran.r-project.org/package=ecr) provides a framework for building evolutionary algorithms for single- and multi-objective continuous or discrete optimization problems.
-   CMA-ES by N. Hansen, global optimization procedure using a covariance matrix adapting evolutionary strategy, is implemented in several packages: In packages [cmaes](https://cran.r-project.org/package=cmaes/index.html) and [cmaesr](../packages/cmaesr/index.html), in [parma](../packages/parma/index.html) as `cmaes`, in [adagio](../packages/adagio/index.html) as `pureCMAES`, and in [rCMA](../packages/rCMA) as `cmaOptimDP`, interfacing Hansen's own Java implementation.
-   Package [Rmalschains](https://cran.r-project.org/package=Rmalschains) implements an algorithm family for continuous optimization called memetic algorithms with local search chains (MA-LS-Chains).
-   An R implementation of the Self-Organising Migrating Algorithm (SOMA) is available in package [soma](https://cran.r-project.org/package=soma). This stochastic optimization method is somewhat similar to genetic algorithms.
-   [nloptr](https://cran.r-project.org/package=nloptr) supports several global optimization routines, such as DIRECT, controlled random search (CRS), multi-level single-linkage (MLSL), improved stochastic ranking (ISR-ES), or stochastic global optimization (StoGO).
-   The [NMOF](https://cran.r-project.org/package=NMOF) package provides implementations of differential evolution, particle swarm optimization, local search and threshold accepting (a variant of simulated annealing). The latter two methods also work for discrete optimization problems, as does the implementation of a genetic algorithm that is included in the package.
-   [SACOBRA](https://cran.r-project.org/package=SACOBRA) is a package for numeric constrained optimization of expensive black-box functions under severely limited budgets; it implements an extension of the COBRA algorithm with initial design generation and self-adjusting random restarts.
-   [RCEIM](https://cran.r-project.org/package=RCEIM) implements a stochastic heuristic method for performing multi-dimensional function optimization.

<span id="mathematical-programming-solvers">Mathematical Programming Solvers</span>
-----------------------------------------------------------------------------------

This section provides an overview of open source as well as commercial optimizers. Which type of mathematical programming problem can be solved by a certain package or function can be seen from the abbreviations in square brackets. For a [Classification According to Subject](#classification-according-to-subject) see the list at the end of this task view.

-   Package [ompr](https://cran.r-project.org/package=ompr) is an optimization modeling package to model and solve Mixed Integer Linear Programs in an algebraic way directly in R. The models are solver-independent and thus offer the possibility to solve models with different solvers. (Inspired by Julia's JuMP project.)
-   [linprog](https://cran.r-project.org/package=linprog/index.html) solves linear programming problems using the function `solveLP()` (the solver is based on [lpSolve](../packages/lpSolve)) and can read model files in MPS format. \[LP\]
-   In the [boot](https://cran.r-project.org/package=boot) package there is a routine called `simplex()` which realizes the two-phase tableau simplex method for (relatively small) linear programming problems. \[LP\]
-   [rcdd](https://cran.r-project.org/package=rcdd) offers the function `lpcdd()` for solving linear programs with exact arithmetic using the [GNU Multiple Precision (GMP)](https://gmplib.org) library. \[LP\]

<!-- -->

-   The [NEOS Server for Optimization](https://www.neos-server.org/neos/) provides online access to state-of-the-art optimization problem solvers. Package [rneos](https://cran.r-project.org/package=rneos) enables the user to pass optimization problems to NEOS and retrieve results within R.

### <span id="interfaces-to-open-source-optimizers">Interfaces to Open Source Optimizers</span>

-   Package [clpAPI](https://cran.r-project.org/package=clpAPI) provides high level access from R to low-level API routines of the [COIN OR Clp](https://projects.coin-or.org/Clp) solver library. \[LP\]
-   Package [lpSolve](https://cran.r-project.org/package=lpSolve/index.html) contains the routine `lp()` to solve LPs and MILPs by calling the freely available solver [lp\_solve](http://lpsolve.sourceforge.net). This solver is based on the revised simplex method and a branch-and-bound (B&B) approach. It supports semi-continuous variables and Special Ordered Sets (SOS). Furthermore `lp.assign()` and `lp.transport()` are aimed at solving assignment problems and transportation problems, respectively. Additionally, there is the package [lpSolveAPI](../packages/lpSolveAPI/index.html) which provides an R interface to the low level API routines of lp\_solve (see also project [<span class="Rforge">lpsolve</span>](https://R-Forge.R-project.org/projects/lpsolve/) on R-Forge). [lpSolveAPI](../packages/lpSolveAPI) supports reading linear programs from files in lp and MPS format. \[BP, IP, LP, MILP, SPLP\]
-   Packages [glpkAPI](https://cran.r-project.org/package=glpkAPI/index.html) as well as package [Rglpk](../packages/Rglpk) provide an interface to the [GNU Linear Programming Kit](https://www.gnu.org/software/glpk/) (GLPK). Whereas the former provides high level access to low level routines the latter offers a high level routine `Rglpk_solve_LP()` to solve MILPs using GLPK. Both packages offer the possibility to use models formulated in the MPS format. \[BP, IP, IPM, LP, MILP\]
-   [Rsymphony](https://cran.r-project.org/package=Rsymphony) has the routine `Rsymphony_solve_LP()` that interfaces the SYMPHONY solver for mixed-integer linear programs. (SYMPHONY is part of the [Computational Infrastructure for Operations Research](http://www.coin-or.org/) (COIN-OR) project.) Package `lsymphony` in Bioconductor provides a similar interface to SYMPHONY that is easier to install. \[LP, IP, MILP\]
-   The NOMAD solver is implemented in the [crs](https://cran.r-project.org/package=crs) package for solving mixed integer programming problems. This algorithm is accessible via the `snomadr()` function and is primarily designed for constrained optimization of blackbox functions.

### <span id="interfaces-to-commercial-optimizers">Interfaces to Commercial Optimizers</span>

This section surveys interfaces to commercial solvers. Typically, the corresponding libraries have to be installed separately.

-   Packages [cplexAPI](https://cran.r-project.org/package=cplexAPI/index.html) and [Rcplex](../packages/Rcplex) provide interfaces to the IBM [CPLEX Optimizer](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/). CPLEX provides dual/primal simplex optimizers as well as a barrier optimizer for solving large scale linear and quadratic programs. It offers a mixed integer optimizer to solve difficult mixed integer programs including (possibly non-convex) MIQCP. Note that CPLEX is **not free** and you have to get a license. Academics will receive a free licence upon request. \[LP, IP, BP, QP, MILP, MIQP, IPM\]
-   The API of the commercial solver LINDO can be accessed in R via package [rLindo](https://cran.r-project.org/package=rLindo). The [LINDO API](http://www.lindo.com/) allows for solving linear, integer, quadratic, conic, general nonlinear, global and stochastic programming problems. \[LP, IP, BP, QP,MILP, MIQP, SP\]
-   Package [Rmosek](https://cran.r-project.org/package=Rmosek) offers an interface to the commercial optimizer from [MOSEK](https://www.mosek.com/). It provides dual/primal simplex optimizers as well as a barrier optimizer. In addition to solving LP and QP problems this solver can handle SOCP and quadratically constrained programming (QPQC) tasks. Furthermore, it offers a mixed integer optimizer to solve difficult mixed integer programs (MILP, MISOCP, etc.). You have to get a license, but Academic licenses are free of charge. \[LP, IP, BP, QP, MILP, MIQP, IPM\]
-   Gurobi Optimization ships an R binding since their 5.0 release that allows to solve LP, MIP, QP, MIQP, SOCP, and MISOCP models from within R. See the [R with Gurobi](https://www.gurobi.com/products/modeling-languages/r) website for more details. \[LP, QP, MILP, MIQP\]
-   The [localsolver](https://cran.r-project.org/package=localsolver) package provides an interface to the hybrid mathematical programming software [LocalSolver](http://www.localsolver.com/) from Innovation 24. LocalSolver is a commercial product, academic licenses are available on request. \[LP, MIP, QP, NLP, HEUR\]

<span id="discrete-optimization-and-graph-theory">Discrete Optimization and Graph Theory</span>
-----------------------------------------------------------------------------------------------

-   Package [adagio](https://cran.r-project.org/package=adagio) provides functions for single and multiple knapsack problems and solves subset sum and assignment tasks.
-   In package [clue](https://cran.r-project.org/package=clue) `solve_LSAP()` enables the user to solve the linear sum assignment problem (LSAP) using an efficient C implementation of the Hungarian algorithm. \[SPLP\]
-   Package [qap](https://cran.r-project.org/package=qap) solves Quadratic Assignment Problems (QAP) applying a simulated annealing heuristics (other approaches will follow).
-   [igraph](https://cran.r-project.org/package=igraph), a package for graph and network analysis, uses the very fast igraph C library. It can be used to calculate shortest paths, maximal network flows, minimum spanning trees, etc. \[GRAPH\]
-   [mknapsack](https://cran.r-project.org/package=mknapsack) solves multiple knapsack problems, based on LP solvers such as 'lpSolve' or 'CBC'; will assign items to knapsacks in a way that the value of the top knapsacks is as large as possible.
-   Package 'knapsack' (see project [<span class="Rforge">optimist</span>](https://R-Forge.R-project.org/projects/optimist/) on R-Forge) provides routines from the book \`Knapsack Problems' by Martello and Toth. There are functions for (multiple) knapsack, subsetsum and binpacking problems. (Use of Fortran codes is restricted to personal research and academic purposes only.)
-   [matchingR](https://cran.r-project.org/package=matchingR/index.html) and [matchingMarkets](../packages/matchingMarkets) implement the Gale-Shapley algorithm for the stable marriage and the college admissions problem, the stable roommates and the house allocation problem. \[COP, MM\]
-   Package [TSP](https://cran.r-project.org/package=TSP/index.html) provides basic infrastructure for handling and solving the traveling salesperson problem (TSP). The main routine `solve_TSP()` solves the TSP through several heuristics. In addition, it provides an interface to the [Concorde TSP Solver](http://www.tsp.gatech.edu/concorde), which has to be downloaded separately. \[SPLP\]

<span id="specific-applications-in-optimization">Specific Applications in Optimization</span>
---------------------------------------------------------------------------------------------

-   Package [nleqslv](https://cran.r-project.org/package=nleqslv) provides function `nleqslv()`, implementing Newton and Broyden methods with line search and trust region global strategies for solving medium sized system of nonlinear equations.
-   Package [goalprog](https://cran.r-project.org/package=goalprog) provides some functions for lexicographic linear goal programming and optimization. Goal programming is a branch of multi-objective, multi-criteria decision analysis. \[MOP\]
-   [mlrMBO](https://cran.r-project.org/package=mlrMBO) is a flexible and comprehensive R toolbox for model-based optimization ('MBO'), also known as Bayesian optimization. It is designed for both single- and multi-objective optimization with mixed continuous, categorical and conditional parameters. \[MOP\]
-   The data cloning algorithm is a global optimization approach and a variant of simulated annealing which has been implemented in package [dclone](https://cran.r-project.org/package=dclone). The package provides low level functions for implementing maximum likelihood estimating procedures for complex models using data cloning and Bayesian Markov chain Monte Carlo methods.
-   [irace](https://cran.r-project.org/package=irace) contains an optimization algorithm for optimizing the parameters of other optimization algorithms. This problem is called "(offline) algorithm configuration". \[GO\]
-   Package [kofnGA](https://cran.r-project.org/package=kofnGA) uses a genetic algorithm to choose a subset of a fixed size k from the integers 1:n, such that a user- supplied objective function is minimized at that subset.
-   [copulaedas](https://cran.r-project.org/package=copulaedas) provides a platform where 'estimation of distribution algorithms' (EDA) based on copulas can be implemented and studied; the package offers various EDAs, and newly developed EDAs can be integrated by extending an S4 class.
-   [tabuSearch](https://cran.r-project.org/package=tabuSearch) implements a tabu search algorithm for optimizing binary strings, maximizing a user defined target function, and returns the best (i.e. maximizing) binary configuration found.
-   Besides functionality for solving general isotone regression problems, package [isotone](https://cran.r-project.org/package=isotone) provides a framework of active set methods for isotone optimization problems with arbitrary order restrictions.
-   Multi-criteria optimization problems can be solved using package [mco](https://cran.r-project.org/package=mco) which implements genetic algorithms. \[MOP\]
-   Package [optmatch](https://cran.r-project.org/package=optmatch) provides routines for solving matching problems by translating them into minimum-cost flow problems, which are in turn solved optimally by the RELAX-IV codes of Bertsekas and Tseng (free for research). \[SPLP\]
-   The [desirability](https://cran.r-project.org/package=desirability) package contains S3 classes for multivariate optimization using the desirability function approach of Harrington (1965) using functional forms described by Derringer and Suich (1980).
-   Package [sna](https://cran.r-project.org/package=sna) contains the function `lab.optim()` which is the front-end to a series of heuristic routines for optimizing some bivariate graph statistic. \[GRAPH\]
-   [maxLik](https://cran.r-project.org/package=maxLik) adds a likelihood-specific layer on top of a number of maximization routines like Brendt-Hall-Hall-Hausman (BHHH) and Newton-Raphson among others. It includes summary and print methods which extract the standard errors based on the Hessian matrix and allows easy swapping of maximization algorithms. It also provides a function to check whether an analytic derivative is computed directly.

<span id="classification-according-to-subject">Classification According to Subject</span>
-----------------------------------------------------------------------------------------

What follows is an attempt to provide a by-subject overview of packages. The full name of the subject as well as the corresponding [MSC 2010](http://www.ams.org/mathscinet/msc/msc2010.html?t=90Cxx&btn=Current) code (if available) are given in brackets.

-   LP (Linear programming, 90C05): [boot](https://cran.r-project.org/package=boot/index.html), [clpAPI](../packages/clpAPI/index.html), [cplexAPI](../packages/cplexAPI/index.html), [glpkAPI](../packages/glpkAPI/index.html), [limSolve](../packages/limSolve/index.html), [linprog](../packages/linprog/index.html), [lpSolve](../packages/lpSolve/index.html), [lpSolveAPI](../packages/lpSolveAPI/index.html), [quantreg](../packages/quantreg/index.html), [rcdd](../packages/rcdd/index.html), [Rcplex](../packages/Rcplex/index.html), [Rglpk](../packages/Rglpk/index.html), [rLindo](../packages/rLindo/index.html), [Rmosek](../packages/Rmosek/index.html), [Rsymphony](../packages/Rsymphony)
-   GO (Global Optimization): [DEoptim](https://cran.r-project.org/package=DEoptim/index.html), [DEoptimR](../packages/DEoptimR/index.html), [GenSA](../packages/GenSA/index.html), [GA](../packages/GA/index.html), [pso](../packages/pso/index.html), [hydroPSO](../packages/hydroPSO/index.html), [cmaes](../packages/cmaes/index.html), [nloptr](../packages/nloptr/index.html), [NMOF](../packages/NMOF)
-   SPLP (Special problems of linear programming like transportation, multi-index, etc., 90C08): [clue](https://cran.r-project.org/package=clue/index.html), [lpSolve](../packages/lpSolve/index.html), [lpSolveAPI](../packages/lpSolveAPI/index.html), [quantreg](../packages/quantreg/index.html), [TSP](../packages/TSP)
-   BP (Boolean programming, 90C09): [cplexAPI](https://cran.r-project.org/package=cplexAPI/index.html), [glpkAPI](../packages/glpkAPI/index.html), [lpSolve](../packages/lpSolve/index.html), [lpSolveAPI](../packages/lpSolveAPI/index.html), [Rcplex](../packages/Rcplex/index.html), [Rglpk](../packages/Rglpk)
-   IP (Integer programming, 90C10): [cplexAPI](https://cran.r-project.org/package=cplexAPI/index.html), [glpkAPI](../packages/glpkAPI/index.html), [lpSolve](../packages/lpSolve/index.html), [lpSolveAPI](../packages/lpSolveAPI/index.html), [Rcplex](../packages/Rcplex/index.html), [Rglpk](../packages/Rglpk/index.html), [rLindo](../packages/rLindo/index.html) [Rmosek](../packages/Rmosek/index.html), [Rsymphony](../packages/Rsymphony)
-   MIP (Mixed integer programming and its variants MILP for LP and MIQP for QP, 90C11): [cplexAPI](https://cran.r-project.org/package=cplexAPI/index.html), [glpkAPI](../packages/glpkAPI/index.html), [lpSolve](../packages/lpSolve/index.html), [lpSolveAPI](../packages/lpSolveAPI/index.html), [Rcplex](../packages/Rcplex/index.html), [Rglpk](../packages/Rglpk/index.html), [rLindo](../packages/rLindo/index.html), [Rmosek](../packages/Rmosek/index.html), [Rsymphony](../packages/Rsymphony)
-   SP (Stochastic programming, 90C15): [rLindo](https://cran.r-project.org/package=rLindo)
-   QP (Quadratic programming, 90C20): [cplexAPI](https://cran.r-project.org/package=cplexAPI/index.html), [kernlab](../packages/kernlab/index.html), [limSolve](../packages/limSolve/index.html), [LowRankQP](../packages/LowRankQP/index.html), [quadprog](../packages/quadprog/index.html), [Rcplex](../packages/Rcplex/index.html), [Rmosek](../packages/Rmosek)
-   SDP (Semidefinite programming, 90C22): [Rcsdp](https://cran.r-project.org/package=Rcsdp/index.html), [Rdsdp](../packages/Rdsdp)
-   CP (Convex programming, 90C25): [cccp](https://cran.r-project.org/package=cccp/index.html), [CLSOCP](../packages/CLSOCP)
-   COP (Combinatorial optimization, 90C27): [adagio](https://cran.r-project.org/package=adagio/index.html), [CEoptim](../packages/CEoptim/index.html), [TSP](../packages/TSP/index.html), [matchingR](../packages/matchingR)
-   MOP (Multi-objective and goal programming, 90C29): [goalprog](https://cran.r-project.org/package=goalprog/index.html), [mco](../packages/mco)
-   NLP (Nonlinear programming, 90C30): [nloptr](https://cran.r-project.org/package=nloptr/index.html), [alabama](../packages/alabama/index.html), [Rsolnp](../packages/Rsolnp/index.html), Rdonlp2, [rLindo](../packages/rLindo)
-   GRAPH (Programming involving graphs or networks, 90C35): [igraph](https://cran.r-project.org/package=igraph/index.html), [sna](../packages/sna)
-   IPM (Interior-point methods, 90C51): [cplexAPI](https://cran.r-project.org/package=cplexAPI/index.html), [kernlab](../packages/kernlab/index.html), [glpkAPI](../packages/glpkAPI/index.html), [LowRankQP](../packages/LowRankQP/index.html), [quantreg](../packages/quantreg/index.html), [Rcplex](../packages/Rcplex)
-   RGA (Methods of reduced gradient type, 90C52): stats ( `optim()`), [gsl](https://cran.r-project.org/package=gsl)
-   QN (Methods of quasi-Newton type, 90C53): stats ( `optim()`), [gsl](https://cran.r-project.org/package=gsl/index.html), [lbfgs](../packages/lbfgs/index.html), [lbfgsb3](../packages/lbfgsb3/index.html), [nloptr](../packages/nloptr/index.html), [ucminf](../packages/ucminf)
-   DF (Derivative-free methods, 90C56): [dfoptim](https://cran.r-project.org/package=dfoptim/index.html), [minqa](../packages/minqa/index.html), [nloptr](../packages/nloptr)
-   HEUR (Approximation methods and heuristics, 90C59): [irace](https://cran.r-project.org/package=irace)

### CRAN packages:

-   [ABCoptim](https://cran.r-project.org/package=ABCoptim)
-   [adagio](https://cran.r-project.org/package=adagio)
-   [alabama](https://cran.r-project.org/package=alabama) (core)
-   [BB](https://cran.r-project.org/package=BB)
-   [boot](https://cran.r-project.org/package=boot)
-   [bvls](https://cran.r-project.org/package=bvls)
-   [cccp](https://cran.r-project.org/package=cccp)
-   [cec2005benchmark](https://cran.r-project.org/package=cec2005benchmark)
-   [cec2013](https://cran.r-project.org/package=cec2013)
-   [CEoptim](https://cran.r-project.org/package=CEoptim)
-   [clpAPI](https://cran.r-project.org/package=clpAPI)
-   [CLSOCP](https://cran.r-project.org/package=CLSOCP)
-   [clue](https://cran.r-project.org/package=clue)
-   [cmaes](https://cran.r-project.org/package=cmaes)
-   [cmaesr](https://cran.r-project.org/package=cmaesr)
-   [colf](https://cran.r-project.org/package=colf)
-   [coneproj](https://cran.r-project.org/package=coneproj)
-   [copulaedas](https://cran.r-project.org/package=copulaedas)
-   [cplexAPI](https://cran.r-project.org/package=cplexAPI)
-   [crs](https://cran.r-project.org/package=crs)
-   [CVXR](https://cran.r-project.org/package=CVXR)
-   [dclone](https://cran.r-project.org/package=dclone)
-   [DEoptim](https://cran.r-project.org/package=DEoptim) (core)
-   [DEoptimR](https://cran.r-project.org/package=DEoptimR)
-   [desirability](https://cran.r-project.org/package=desirability)
-   [dfoptim](https://cran.r-project.org/package=dfoptim) (core)
-   [Dykstra](https://cran.r-project.org/package=Dykstra)
-   [ECOSolveR](https://cran.r-project.org/package=ECOSolveR)
-   [ecr](https://cran.r-project.org/package=ecr)
-   [flacco](https://cran.r-project.org/package=flacco)
-   [GA](https://cran.r-project.org/package=GA)
-   [genalg](https://cran.r-project.org/package=genalg)
-   [GenSA](https://cran.r-project.org/package=GenSA)
-   [globalOptTests](https://cran.r-project.org/package=globalOptTests)
-   [glpkAPI](https://cran.r-project.org/package=glpkAPI)
-   [goalprog](https://cran.r-project.org/package=goalprog)
-   [GrassmannOptim](https://cran.r-project.org/package=GrassmannOptim)
-   [gsl](https://cran.r-project.org/package=gsl)
-   [hydroPSO](https://cran.r-project.org/package=hydroPSO)
-   [igraph](https://cran.r-project.org/package=igraph)
-   [irace](https://cran.r-project.org/package=irace)
-   [isotone](https://cran.r-project.org/package=isotone)
-   [kernlab](https://cran.r-project.org/package=kernlab)
-   [kofnGA](https://cran.r-project.org/package=kofnGA)
-   [lbfgs](https://cran.r-project.org/package=lbfgs)
-   [lbfgsb3](https://cran.r-project.org/package=lbfgsb3)
-   [limSolve](https://cran.r-project.org/package=limSolve)
-   [linprog](https://cran.r-project.org/package=linprog)
-   [localsolver](https://cran.r-project.org/package=localsolver)
-   [LowRankQP](https://cran.r-project.org/package=LowRankQP)
-   [lpSolve](https://cran.r-project.org/package=lpSolve)
-   [lpSolveAPI](https://cran.r-project.org/package=lpSolveAPI)
-   [lsei](https://cran.r-project.org/package=lsei)
-   [ManifoldOptim](https://cran.r-project.org/package=ManifoldOptim)
-   [matchingMarkets](https://cran.r-project.org/package=matchingMarkets)
-   [matchingR](https://cran.r-project.org/package=matchingR)
-   [maxLik](https://cran.r-project.org/package=maxLik)
-   [mcga](https://cran.r-project.org/package=mcga)
-   [mco](https://cran.r-project.org/package=mco)
-   [metaheuristicOpt](https://cran.r-project.org/package=metaheuristicOpt)
-   [minpack.lm](https://cran.r-project.org/package=minpack.lm)
-   [minqa](https://cran.r-project.org/package=minqa)
-   [mize](https://cran.r-project.org/package=mize)
-   [mknapsack](https://cran.r-project.org/package=mknapsack)
-   [mlrMBO](https://cran.r-project.org/package=mlrMBO)
-   [n1qn1](https://cran.r-project.org/package=n1qn1)
-   [neldermead](https://cran.r-project.org/package=neldermead)
-   [NlcOptim](https://cran.r-project.org/package=NlcOptim)
-   [nleqslv](https://cran.r-project.org/package=nleqslv)
-   [nlmrt](https://cran.r-project.org/package=nlmrt)
-   [nloptr](https://cran.r-project.org/package=nloptr)
-   [nls2](https://cran.r-project.org/package=nls2)
-   [nlsr](https://cran.r-project.org/package=nlsr)
-   [NMOF](https://cran.r-project.org/package=NMOF)
-   [nnls](https://cran.r-project.org/package=nnls)
-   [ompr](https://cran.r-project.org/package=ompr)
-   [onls](https://cran.r-project.org/package=onls)
-   [optimr](https://cran.r-project.org/package=optimr)
-   [optimsimplex](https://cran.r-project.org/package=optimsimplex)
-   [optimx](https://cran.r-project.org/package=optimx)
-   [optmatch](https://cran.r-project.org/package=optmatch)
-   [parma](https://cran.r-project.org/package=parma)
-   [powell](https://cran.r-project.org/package=powell)
-   [pso](https://cran.r-project.org/package=pso)
-   [psoptim](https://cran.r-project.org/package=psoptim)
-   [qap](https://cran.r-project.org/package=qap)
-   [quadprog](https://cran.r-project.org/package=quadprog) (core)
-   [quadprogXT](https://cran.r-project.org/package=quadprogXT)
-   [quantreg](https://cran.r-project.org/package=quantreg)
-   [rcdd](https://cran.r-project.org/package=rcdd)
-   [RCEIM](https://cran.r-project.org/package=RCEIM)
-   [Rcgmin](https://cran.r-project.org/package=Rcgmin)
-   [rCMA](https://cran.r-project.org/package=rCMA)
-   [Rcplex](https://cran.r-project.org/package=Rcplex)
-   [RcppDE](https://cran.r-project.org/package=RcppDE)
-   [RcppNumerical](https://cran.r-project.org/package=RcppNumerical)
-   [Rcsdp](https://cran.r-project.org/package=Rcsdp)
-   [Rdsdp](https://cran.r-project.org/package=Rdsdp)
-   [rgenoud](https://cran.r-project.org/package=rgenoud)
-   [Rglpk](https://cran.r-project.org/package=Rglpk)
-   [rLindo](https://cran.r-project.org/package=rLindo)
-   [Rmalschains](https://cran.r-project.org/package=Rmalschains)
-   [Rmosek](https://cran.r-project.org/package=Rmosek)
-   [rneos](https://cran.r-project.org/package=rneos)
-   [ROI](https://cran.r-project.org/package=ROI)
-   [ROI.plugin.qpoases](https://cran.r-project.org/package=ROI.plugin.qpoases)
-   [rosqp](https://cran.r-project.org/package=rosqp)
-   [Rsolnp](https://cran.r-project.org/package=Rsolnp)
-   [Rsymphony](https://cran.r-project.org/package=Rsymphony)
-   [Rtnmin](https://cran.r-project.org/package=Rtnmin)
-   [Rvmmin](https://cran.r-project.org/package=Rvmmin)
-   [SACOBRA](https://cran.r-project.org/package=SACOBRA)
-   [scs](https://cran.r-project.org/package=scs)
-   [sdpt3r](https://cran.r-project.org/package=sdpt3r)
-   [smoof](https://cran.r-project.org/package=smoof)
-   [sna](https://cran.r-project.org/package=sna)
-   [soma](https://cran.r-project.org/package=soma)
-   [subplex](https://cran.r-project.org/package=subplex)
-   [tabuSearch](https://cran.r-project.org/package=tabuSearch)
-   [trust](https://cran.r-project.org/package=trust)
-   [trustOptim](https://cran.r-project.org/package=trustOptim)
-   [TSP](https://cran.r-project.org/package=TSP)
-   [ucminf](https://cran.r-project.org/package=ucminf) (core)

### Related links:

-   [Journal of Statistical Software Special Volume on Optimization (Editor: Ravi Varadhan)](https://www.jstatsoft.org/v60)
-   [Nonlinear Parameter Optimization Using R Tools -- John C. Nash (Wiley)](http://www.wiley.com/WileyCDA/WileyTitle/productCd-1118569288.html)
-   [Modern Optimization With R -- Paulo Cortez (Springer UseR Series)](https://www.springer.com/mathematics/book/978-3-319-08262-2)
-   [COIN-OR Project](http://www.coin-or.org/)
-   [NEOS Optimization Guide](http://www.neos-guide.org/Optimization-Guide)
-   [Decision Tree for Optimization Software](http://plato.asu.edu/sub/pns.html)
-   [Mathematics Subject Classification - Mathematical programming](http://www.ams.org/mathscinet/msc/msc2010.html?t=90Cxx&btn=Current)
