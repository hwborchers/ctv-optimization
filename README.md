
# Optimization and Mathematical Programming

|                 |                                                        |
|-----------------|--------------------------------------------------------|  
| **Maintainer:** | Hans W. Borchers                                       | 
| **Contact:**    | hwb at mailbox.org                                     | 
| **Version:**    | 2023-08-06                                             | 
| **Web page**    | [Optimization and Mathematical Programming](http://htmlpreview.github.io/?https://github.com/hwborchers/ctv-optimization/blob/master/optimization.html) |

This page has been forged from the CRAN Task View "Optimization and Mathematical Programming", version end of 2021.
For the current CRAN Task View see [here](https://CRAN.R-project.org/view=Optimization). 
This page will be further developed in a different direction, hoping
to give the user more extensive and useful information.
It still contains a list of R packages that offer facilities
for solving numerical and combinatorial optimization problems,
including statistical regression tasks modeled as optimization problems. 

**Contents**

* [Optimization Infrastructure Packages](#infrastr)
* [General Purpose Continuous Solvers](#general)
* [Quadratic Optimization](#quadratic)
* [Test and Benchmarking Collections](#benchmark)
* [Least-Squares Problems](#leastsquares)
* [Semidefinite and Convex Solvers](#convex)
* [Global and Stochastic Optimization](#global)
* [Mathematical Programming Solvers](#mathprog)
* [Combinatorial Optimization](#discrete)
* [Multi Objective Optimization](#multiobj)

Packages in this view are roughly structured according to these topics.
(See also the "Related links" section at the end of the task view.)
Please note that many packages provide functionality for more than one 
class of optimization problems. Suggestions and improvements for this task
view are welcome and can be made through issues or pull requests on GitHub
or via e-mail to the maintainer address.


### [Optimization Infrastructure Packages]{#infrastr}

-   The [optimx](https://cran.r-project.org/web/packages/optimx/index.html) package provides a replacement and
    extension of the `optim()` function in Base R with a call to several
    function minimization codes in R in a single statement. These
    methods handle smooth, possibly box-constrained functions of several
    or many parameters. Function `optimr()` in this package extends the
    `optim()` function with the same syntax but more 'method' choices.
    Function `opm()` applies several solvers to a selected optimization
    task and returns a data frame of results for easy comparison.

-   The R Optimization Infrastructure ([ROI](https://cran.r-project.org/web/packages/ROI/index.html)) package
    provides a framework for handling optimization problems in R. It
    uses an object-oriented approach to define and solve various
    optimization tasks from different problem classes (e.g., linear,
    quadratic, non-linear programming problems). This makes optimization
    transparent for the user as the corresponding workflow is abstracted
    from the underlying solver. The approach allows for easy switching
    between solvers and thus enhances comparability. For more
    information see the [ROI home page](http://roi.r-forge.r-project.org/).

-   The package [CVXR](https://cran.r-project.org/web/packages/CVXR/index.html) provides an object-oriented
    modeling language for Disciplined Convex Programming (DCP). It
    allows the user to formulate convex optimization problems in a
    natural way following mathematical convention and DCP rules. The
    system analyzes the problem, verifies its convexity, converts it
    into a canonical form, and hands it off to an appropriate solver
    such as ECOS or SCS to obtain the solution.For more information 
    see the [CVXR home page](https://cvxr.rbind.io/).


### [General Purpose Continuous Solvers]{#general}

Package stats offers several general-purpose optimization routines. For
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

-   Package [lbfgs](https://cran.r-project.org/web/packages/lbfgs/index.html) wraps the libBFGS C library by
    Okazaki and Morales (converted from Nocedal's L-BFGS-B 3.0 Fortran
    code), interfacing both the L-BFGS and the OWL-QN algorithm, the
    latter being particularly suited for higher-dimensional problems.
-   [lbfgsb3c](https://cran.r-project.org/web/packages/lbfgsb3c/index.html) interfaces J.Nocedal's L-BFGS-B 3.0
    Fortran code, a limited memory BFGS minimizer, allowing bound
    constraints and is applicable to higher-dimensional problems. It
    has an 'optim'-like interface based on 'Rcpp'.
-   Package [roptim](https://cran.r-project.org/web/packages/roptim/index.html) provides a unified wrapper to
    call C++ functions of the algorithms underlying the optim() solver;
    and [optimParallel](https://cran.r-project.org/web/packages/optimParallel/index.html) provides a parallel version
    of the L-BFGS-B method of optim(); using these packages can
    significantly reduce the optimization time.
-   [RcppNumerical](https://cran.r-project.org/web/packages/RcppNumerical/index.html) is a collection of open-source
    libraries for numerical computing and their integration with
    'Rcpp'. It provides a wrapper for the L-BFGS algorithm, based on
    the LBFGS++ library (based on code of N. Okazaki).
-   Package [ucminf](https://cran.r-project.org/web/packages/ucminf/index.html) implements an
    algorithm of quasi-Newton type for nonlinear unconstrained
    optimization, combining a trust region with line search approaches.
    The interface of `ucminf()` is designed for easy interchange with
    `optim()`.
-   The following packages implement optimization routines in pure R,
    for nonlinear functions with bounds constraints:
    [Rcgmin](https://cran.r-project.org/web/packages/Rcgmin/index.html): gradient function minimization similar
    to GC; [Rvmmin](https://cran.r-project.org/web/packages/Rvmmin/index.html): variable metric function
    minimization; [Rtnmin](https://cran.r-project.org/web/packages/Rtnmin/index.html): truncated Newton function
    minimization.
-   [marqLevAlg](https://cran.r-project.org/web/packages/marqLevAlg/index.html)
    implements a parallelized version of the Marquardt-Levenberg algorithm.
    It is particularly suited for complex problems and when starting from points very far from the final optimum.
    The package is designed to be used for unconstrained local optimization.
-   [mize](https://cran.r-project.org/web/packages/mize/index.html) implements optimization algorithms in pure
    R, including conjugate gradient (CG),
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) and limited memory BFGS
    (L-BFGS) methods. Most internal parameters can be set through the
    calling interface.
-   [n1qn1](https://cran.r-project.org/web/packages/n1qn1/index.html) provides an R port of the `n1qn1` optimization procedure 
    ported from Scilab, a quasi-Newton BFGS method without constraints.
-   [stochQN](https://cran.r-project.org/web/packages/stochQN/index.html) provides implementations of stochastic,
    limited-memory quasi-Newton optimizers, similar in spirit to the
    LBFGS. It includes an implementation of online LBFGS, stochastic
    quasi-Newton and adaptive quasi-Newton.
-   [nonneg.cg](https://cran.r-project.org/web/packages/nonneg.cg/index.html) realizes a conjugate-gradient based
    method to minimize functions subject to all variables being
    non-negative.
-   Package [dfoptim](https://cran.r-project.org/web/packages/dfoptim/index.html), for
    derivative-free optimization procedures, contains quite efficient R
    implementations of the Nelder-Mead and Hooke-Jeeves algorithms
    (unconstrained and with bounds constraints).
-   Package [nloptr](https://cran.r-project.org/web/packages/nloptr/index.html) provides access to NLopt, an
    LGPL licensed library of various nonlinear optimization algorithms.
    It includes local derivative-free (COBYLA, Nelder-Mead, Subplex) and
    gradient-based (e.g., BFGS) methods, and also the augmented
    Lagrangian approach for nonlinear constraints.
-   Package [alabama](https://cran.r-project.org/web/packages/alabama/index.html) provides an
    implementations of the Augmented Lagrange Barrier minimization
    algorithm for optimizing smooth nonlinear objective functions with
    (nonlinear) equality and inequality constraints.
-   Package [Rsolnp](https://cran.r-project.org/web/packages/Rsolnp/index.html) provides an implementation of
    the Augmented Lagrange Multiplier method for solving nonlinear
    optimization problems with equality and inequality constraints
    (based on code by Y. Ye).
-   [NlcOptim](https://cran.r-project.org/web/packages/NlcOptim/index.html) solves nonlinear optimization problems
    with linear and nonlinear equality and inequality constraints,
    implementing a Sequential Quadratic Programming (SQP) method;
    accepts the input parameters as a constrained matrix.
-   In package Rdonlp2 (see the `r rforge("rmetrics")`
    project) function `donlp2()`, a wrapper for the DONLP2 solver,
    offers the minimization of smooth nonlinear functions and
    constraints. DONLP2 can be used freely for any kind of research
    purposes, otherwise it requires licensing.
-   [psqn](https://cran.r-project.org/web/packages/psqn/index.html) provides quasi-Newton methods to minimize
    partially separable functions; the methods are largely described in
    "Numerical Optimization" by Nocedal and Wright (2006).
-   [clue](https://cran.r-project.org/web/packages/clue/index.html) contains the function `sumt()` for solving
    constrained optimization problems via the sequential unconstrained
    minimization technique (SUMT).
-   [BB](https://cran.r-project.org/web/packages/BB/index.html) contains the function `spg()` providing a
    spectral projected gradient method for large-scale optimization with
    simple constraints. It takes a nonlinear objective function as an
    argument as well as basic constraints.
-   [GrassmannOptim](https://cran.r-project.org/web/packages/GrassmannOptim/index.html) is a package for Grassmann
    manifold optimization. The implementation uses gradient-based
    algorithms and embeds a stochastic gradient method for global
    search.
-   [ManifoldOptim](https://cran.r-project.org/web/packages/ManifoldOptim/index.html) is an R interface to the
    'ROPTLIB' optimization library. It optimizes real-valued functions
    over manifolds such as Stiefel, Grassmann, and Symmetric Positive
    Definite matrices.
-   Function `multimin()` in the [gsl](https://cran.r-project.org/web/packages/gsl/index.html)
    package, based on the GNU Scientific Library ([GSL](https://www.gnu.org/software/gsl/)),
    provides BFGS, conjugate gradient, steepest descent, and Nelder-Mead algorithms.
    NOTE: `multimin()` has been removed from the package temporarily, awaiting a permanent fix.
-   Several derivative-free optimization algorithms are provided with
    package [minqa](https://cran.r-project.org/web/packages/minqa/index.html);
    e.g., the functions `bobyqa()`, newuoa()`, and `uobyqa()` allow minimizing a function of many
    variables by a trust region method that forms quadratic models by interpolation.
    `bobyqa()` additionally permits box constraints bounds) on the parameters.
-   [subplex](https://cran.r-project.org/web/packages/subplex/index.html) provides unconstrained function
    optimization based on a subspace searching simplex method.
-   In package [trust](https://cran.r-project.org/web/packages/trust/index.html), a routine with the same name
    offers local optimization based on the "trust region" approach.
-   [trustOptim](https://cran.r-project.org/web/packages/trustOptim/index.html) implements "trust region" for
    unconstrained nonlinear optimization. The algorithm is optimized for
    objective functions with sparse Hessians.
-   Package [quantreg](https://cran.r-project.org/web/packages/quantreg/index.html) contains variations of simplex
    and of interior point routines ( `nlrq()`, `crq()`). It provides an
    interface to L1 regression in the R code of function `rq()`.


### [Quadratic Optimization]{#quadratic}

-   In package [quadprog](https://cran.r-project.org/web/packages/quadprog/index.html)
    `solve.QP()` solves quadratic programming problems with linear
    equality and inequality constraints. (The matrix has to be positive
    definite.) [quadprogXT](https://cran.r-project.org/web/packages/quadprogXT/index.html) extends this with
    absolute value constraints and absolute values in the objective
    function.
-   [osqp](https://cran.r-project.org/web/packages/osqp/index.html) provides bindings to
    [OSQP](https://osqp.org), the 'Operator Splitting QP' solver from
    the University of Oxford Control Group; it solves sparse convex
    quadratic programming problems with optional equality and inequality
    constraints efficiently.
-   [qpmadr](https://cran.r-project.org/web/packages/qpmadr/index.html) interfaces the 'qpmad' software and
    solves quadratic programming (QP) problems with linear inequality,
    equality and bound constraints, using the method by Goldfarb and
    Idnani.
-   [kernlab](https://cran.r-project.org/web/packages/kernlab/index.html) contains the function `ipop` for
    solving quadratic programming problems using interior point methods.
    (The matrix can be positive semidefinite.)
-   [Dykstra](https://cran.r-project.org/web/packages/Dykstra/index.html) solves quadratic programming problems
    using R. L. Dykstra's cyclic projection algorithm for positive
    definite and semidefinite matrices. The routine allows for a
    combination of equality and inequality constraints.
-   [coneproj](https://cran.r-project.org/web/packages/coneproj/index.html) contains routines for cone projection
    and quadratic programming, estimation, and inference for constrained
    parametric regression, and shape-restricted regression problems.
-   [LowRankQP](https://cran.r-project.org/web/packages/LowRankQP/index.html) (archived) primal/dual
    interior point method for solving quadratic programming problems (especially for semidefinite quadratic forms).
-   The COIN-OR project 'qpOASES' implements a reliable QP solver,
    even when tackling semi-definite or degenerated QP problems; it is
    particularly suited for model predictive control (MPC) applications;
    the ROI plugin [ROI.plugin.qpoases](https://cran.r-project.org/web/packages/ROI.plugin.qpoases/index.html) makes it
    accessible for R users.
-   [mixsqp](https://cran.r-project.org/web/packages/mixsqp/index.html) implements the "mix-SQP" algorithm,
    based on sequential quadratic programming (SQP), for maximum
    likelihood estimations in finite mixture models.
-   [limSolve](https://cran.r-project.org/web/packages/limSolve/index.html) offers to solve linear or quadratic
    optimization functions, subject to equality and/or inequality constraints.
-   [CGNM](https://cran.r-project.org/web/packages/CGNM/index.html) finds multiple solutions of nonlinear
    least-squares problems, without assuming uniqueness of the solution.


### [Test and Benchmarking Collections]{#benchmark}

-   Objective functions for benchmarking the performance of global
    optimization algorithms can be found in
    [globalOptTests](https://cran.r-project.org/web/packages/globalOptTests/index.html).
-   [smoof](https://cran.r-project.org/web/packages/smoof/index.html) has generators for a number of both
    single- and multi-objective test functions that are frequently used
    for benchmarking optimization algorithms; offers a set of convenient
    functions to generate, plot, and work with objective functions.
-   [flacco](https://cran.r-project.org/web/packages/flacco/index.html) contains tools and features used for an
    Exploratory Landscape Analysis (ELA) of continuous optimization
    problems, capable of quantifying rather complex properties, such as
    the global structure, separability, etc., of the optimization
    problems.
-   Package `r github("jlmelville/funconstrain")` (on Github)
    implements 35 of the test functions by More, Garbow, and Hillstom,
    useful for testing unconstrained optimization methods.


### [Least-Squares Problems]{#leastsquares}

Function `solve.qr()` (resp. `qr.solve()`) handles over- and
under-determined systems of linear equations, returning least-squares
solutions if possible. And package stats provides `nls()` to determine
least-squares estimates of the parameters of a nonlinear model.
[nls2](https://cran.r-project.org/web/packages/nls2/index.html) enhances function `nls()` with brute force or
grid-based searches, to avoid being dependent on starting parameters or
getting stuck in local solutions.

-   Package [nlsr](https://cran.r-project.org/web/packages/nlsr/index.html) provides tools for working with
    nonlinear least-squares problems. Functions `nlfb` and `nlxb` are
    intended to eventually supersede the 'nls()' function in Base R,
    by applying a variant of the Marquardt procedure for nonlinear
    least-squares, with bounds constraints and optionally Jacobian
    described as R functions.
-   Package [minpack.lm](https://cran.r-project.org/web/packages/minpack.lm/index.html) provides a function
    `nls.lm()` for solving nonlinear least-squares problems by a
    modification of the Levenberg-Marquardt algorithm, with support for
    lower and upper parameter bounds, as found in MINPACK.
-   Package [onls](https://cran.r-project.org/web/packages/onls/index.html)
    fits two-dimensional data by means of orthogonal
    nonlinear least-squares regression (ONLS), using Levenberg-Marquardt
    minimization; it provides functionality for fit diagnostics and plotting
    and comes into question when one encounters "error in variables" problems.
-   Package [nnls](https://cran.r-project.org/web/packages/nnls/index.html) interfaces the Lawson-Hanson
    implementation of an algorithm for non-negative least-squares,
    allowing the combination of non-negative and non-positive
    constraints.
-   Package [lsei](https://cran.r-project.org/web/packages/lsei/index.html) contains functions that solve least-squares 
    linear regression problems under linear equality/inequality constraints. 
    Functions for solving quadratic programming problems are also available, 
    which transform such problems into least squares ones first. (Based on 
    Fortran programs of Lawson and Hanson.)
-   Package [gslnls](https://cran.r-project.org/web/packages/gslnls/index.html) provides an interface to
    nonlinear least-squares optimization methods from the GNU Scientific
    Library (GSL). The available trust region methods include the
    Levenberg-Marquadt algorithm with and without geodesic acceleration,
    and several variants of Powell's dogleg algorithm.
-   Package [bvls](https://cran.r-project.org/web/packages/bvls/index.html) interfaces the Stark-Parker
    implementation of an algorithm for least-squares with upper and
    lower bounded variables.
-   Package [onls](https://cran.r-project.org/web/packages/onls/index.html) (archived) implements orthogonal
    nonlinear least-squares regression (ONLS, a.k.a. Orthogonal Distance Regression, ODR)
    using a Levenberg-Marquardt-type minimization algorithm based on the ODRPACK Fortran library.
-   [colf](https://cran.r-project.org/web/packages/colf/index.html) performs least squares constrained
    optimization on a linear objective function. It contains a number of
    algorithms to choose from and offers a formula syntax similar to
    `lm()`.


### [Semidefinite and Convex Solvers]{#convex}

-   Package [ECOSolveR](https://cran.r-project.org/web/packages/ECOSolveR/index.html) provides an interface to the
    Embedded COnic Solver (ECOS), a well-known, efficient, and robust C
    library for convex problems. Conic and equality constraints can be
    specified in addition to integer and boolean variable constraints
    for mixed-integer problems.
-   Package [scs](https://cran.r-project.org/web/packages/scs/index.html) applies operator splitting
    to solve linear programs (LPs), second-order cone programs (SOCP),
    semidefinite programs, (SDPs), exponential cone programs (ECPs), and
    power cone programs (PCPs), or problems with any combination of
    those cones.
-   Package [clarabel](https://cran.r-project.org/web/packages/clarabel/index.html)
    provides an interior point numerical solver for convex optimization problems using 
    a novel homogeneous embedding, that solves linear programs (LPs), quadratic programs (QPs),
    second-order cone programs (SOCPs), semidefinite programs (SDPs), and problems with
    exponential and power cone constraints.
    (See Clarabel [Docs](https://oxfordcontrol.github.io/ClarabelDocs/stable/))
-   [sdpt3r](https://cran.r-project.org/web/packages/sdpt3r/index.html) solves general semidefinite Linear
    Programming problems, using an R implementation of the MATLAB
    toolbox SDPT3. Includes problems such as the nearest correlation
    matrix, D-optimal experimental design, Distance Weighted
    Discrimination, or the maximum cut problem.
-   [cccp](https://cran.r-project.org/web/packages/cccp/index.html) contains routines for solving
    cone-constrained convex problems by means of interior-point methods. The
    implemented algorithms are partially ported from CVXOPT, a Python
    module for convex optimization
-   CSDP is a library of routines that implements a primal-dual barrier
    method for solving semidefinite programming problems; it is
    interfaced in the [Rcsdp](https://cran.r-project.org/web/packages/Rcsdp/index.html) package.
-   The DSDP library implements an interior-point method for
    semidefinite programming with primal and dual solutions; it is
    interfaced in package [Rdsdp](https://cran.r-project.org/web/packages/Rdsdp/index.html).


### [Global and Stochastic Optimization]{#global}

-   Package [DEoptim](https://cran.r-project.org/web/packages/DEoptim/index.html) provides a
    global optimizer based on the Differential Evolution algorithm.
    [RcppDE](https://cran.r-project.org/web/packages/RcppDE/index.html) provides a C++ implementation (using
    Rcpp) of the same `DEoptim()` function.
-   [DEoptimR](https://cran.r-project.org/web/packages/DEoptimR/index.html) provides an implementation of the jDE
    variant of the differential evolution stochastic algorithm for
    nonlinear programming problems (It allows handling constraints in a fexible manner.)
-   The [CEoptim](https://cran.r-project.org/web/packages/CEoptim/index.html) package implements a cross-entropy
    optimization technique that can be applied to continuous, discrete,
    mixed, and constrained optimization problems.
-   [GenSA](https://cran.r-project.org/web/packages/GenSA/index.html) is a package providing a function for
    generalized Simulated Annealing which can be used to search for the
    global minimum of a quite complex non-linear objective function with
    a large number of optima.
-   [GA](https://cran.r-project.org/web/packages/GA/index.html) provides functions for optimization using
    Genetic Algorithms in both, the continuous and discrete case. This
    package allows to run corresponding optimization tasks in parallel.
-   In package [gafit](https://cran.r-project.org/web/packages/gafit/index.html) `gafit()` uses a genetic algorithm approach
    to find the minimum of a one-dimensional function.
-   Package [genalg](https://cran.r-project.org/web/packages/genalg/index.html) contains `rbga()`, an implementation
    of a genetic algorithm for multi-dimensional function optimization.
-   Package [rgenoud](https://cran.r-project.org/web/packages/rgenoud/index.html) offers `genoud()`, a routine
    which is capable of solving complex function
    minimization/maximization problems by combining evolutionary
    algorithms with a derivative-based (quasi-Newtonian) approach.
-   Machine coded genetic algorithm (MCGA) provided by package
    [mcga](https://cran.r-project.org/web/packages/mcga/index.html) is a tool that solves optimization
    problems based on byte representation of variables.
-   A particle swarm optimizer (PSO) is implemented in package
    [pso](https://cran.r-project.org/web/packages/pso/index.html), and also in [psoptim](https://cran.r-project.org/web/packages/psoptim/index.html).
    Another (parallelized) implementation of the PSO algorithm can be
    found in package `ppso` available from
    [rforge.net/ppso](https://www.rforge.net/ppso/) .
-   Package [hydroPSO](https://cran.r-project.org/web/packages/hydroPSO/index.html) implements the Standard
    Particle Swarm Optimization (SPSO) algorithm; it is parallel-capable
    and includes several fine-tuning options and post-processing
    functions.
-   `r github("floybix/hydromad")` (on Github) contains the
    `SCEoptim` function for Shuffled Compex Evolution (SCE)
    optimization, an evolutionary algorithm, combined with a simplex
    method.
-   Package [ABCoptim](https://cran.r-project.org/web/packages/ABCoptim/index.html) implements the Artificial Bee
    Colony (ABC) optimization approach.
-   Package [metaheuristicOpt](https://cran.r-project.org/web/packages/metaheuristicOpt/index.html) contains
    implementations of several evolutionary optimization algorithms,
    such as particle swarm, dragonfly and firefly, sine cosine
    algorithms and many others.
-   Package [ecr](https://cran.r-project.org/web/packages/ecr/index.html) provides a framework for building
    evolutionary algorithms for single- and multi-objective continuous
    or discrete optimization problems. And [emoa](https://cran.r-project.org/web/packages/emoa/index.html) has
    a collection of building blocks for the design and analysis of
    evolutionary multiobjective optimization algorithms.
-   CMA-ES by N. Hansen, global optimization procedure using a
    covariance matrix adapting evolutionary strategy, is implemented in
    several packages: In packages [cmaes](https://cran.r-project.org/web/packages/cmaes/index.html) and
    [cmaesr](https://cran.r-project.org/web/packages/cmaesr/index.html), in [parma](https://cran.r-project.org/web/packages/parma/index.html) as
    `cmaes`, in [adagio](https://cran.r-project.org/web/packages/adagio/index.html) as `pureCMAES`, and in
    [rCMA](https://cran.r-project.org/web/packages/rCMA/index.html) as `cmaOptimDP`, interfacing Hansen's own
    Java implementation.
-   Package [Rmalschains](https://cran.r-project.org/web/packages/Rmalschains/index.html) implements an algorithm
    family for continuous optimization called memetic algorithms with
    local search chains (MA-LS-Chains).
-   An R implementation of the Self-Organising Migrating Algorithm
    (SOMA) is available in package [soma](https://cran.r-project.org/web/packages/soma/index.html). This
    stochastic optimization method is somewhat similar to genetic
    algorithms.
-   [nloptr](https://cran.r-project.org/web/packages/nloptr/index.html) supports several global optimization
    routines, such as DIRECT, controlled random search (CRS),
    multi-level single-linkage (MLSL), improved stochastic ranking
    (ISR-ES), or stochastic global optimization (StoGO).
-   The [NMOF](https://cran.r-project.org/web/packages/NMOF/index.html) package provides implementations of
    differential evolution, particle swarm optimization, local search
    and threshold accepting (a variant of simulated annealing). The
    latter two methods also work for discrete optimization problems, as
    does the implementation of a genetic algorithm that is included in
    the package.
-   [SACOBRA](https://cran.r-project.org/web/packages/SACOBRA/index.html) is a package for numeric constrained
    optimization of expensive black-box functions under severely limited
    budgets; it implements an extension of the COBRA algorithm with
    initial design generation and self-adjusting random restarts.
-   [OOR](https://cran.r-project.org/web/packages/OOR/index.html) implements optimistic optimization methods
    for global optimization of deterministic or stochastic functions.
-   [RCEIM](https://cran.r-project.org/web/packages/RCEIM/index.html) implements a stochastic heuristic method
    for performing multi-dimensional function optimization.
-   Package [graDiEnt](https://cran.r-project.org/web/packages/graDiEnt/index.html) implements the Stochastic
    Quasi-Gradient Differential Evolution (SQG-DE) optimization algorithm; being derivative-free, it combines the
    robustness of the population-based "Differential Evolution" with the efficiency of gradient-based optimization.


### [Mathematical Programming Solvers]{#mathprog}

This section provides an overview of open source as well as commercial
optimizers.

-   Package [ompr](https://cran.r-project.org/web/packages/ompr/index.html) is an optimization modeling
    package to model and solve Mixed Integer Linear Programs in an
    algebraic way directly in R. The models are solver-independent and
    thus offer the possibility to solve models with different solvers.
    (Inspired by Julia's JuMP project.)
-   [linprog](https://cran.r-project.org/web/packages/linprog/index.html) solves linear programming problems
    using the function `solveLP()` (the solver is based on
    [lpSolve](https://cran.r-project.org/web/packages/lpSolve/index.html)) and can read model files in MPS
    format.
-   In the [boot](https://cran.r-project.org/web/packages/boot/index.html) package there is a routine called
    `simplex()` which realizes the two-phase tableau simplex method for
    (relatively small) linear programming problems.
-   [rcdd](https://cran.r-project.org/web/packages/rcdd/index.html) offers the function `lpcdd()` for solving
    linear programs with exact arithmetic using the [GNU Multiple
    Precision (GMP)](https://gmplib.org) library.
-   The [NEOS Server for
    Optimization](https://www.neos-server.org/neos/) provides online
    access to state-of-the-art optimization problem solvers. The
    packages [rneos](https://cran.r-project.org/web/packages/rneos/index.html) and [ROI.plugin.neos](https://cran.r-project.org/web/packages/ROI.plugin.neos/index.html) enable
    the user to pass optimization problems to NEOS and retrieve results
    within R.

#### Interfaces to Open Source Optimizers

-   Package [lpSolve](https://cran.r-project.org/web/packages/lpSolve/index.html) contains the routine `lp()` to
    solve LPs and MILPs by calling the freely available solver
    [lp_solve](http://lpsolve.sourceforge.net). This solver is based on
    the revised simplex method and a branch-and-bound (B&B) approach. It
    supports semi-continuous variables and Special Ordered Sets (SOS).
    Furthermore `lp.assign()` and `lp.transport()` are aimed at solving
    assignment problems and transportation problems, respectively.
    Additionally, there is the package [lpSolveAPI](https://cran.r-project.org/web/packages/lpSolveAPI/index.html)
    which provides an R interface to the low-level API routines of
    lp_solve (see also project `r rforge("lpsolve")` on
    R-Forge). [lpSolveAPI](https://cran.r-project.org/web/packages/lpSolveAPI/index.html) supports reading linear
    programs from files in lp and MPS format.
-   Packages [glpkAPI](https://cran.r-project.org/web/packages/glpkAPI/index.html) as well as package
    [Rglpk](https://cran.r-project.org/web/packages/Rglpk/index.html) provide an interface to the [GNU Linear
    Programming Kit](https://www.gnu.org/software/glpk/) (GLPK). Whereas
    the former provides access to low-level routines the, latter provides a routine `Rglpk_solve_LP()` to solve MILPs
    using GLPK. Both packages offer the possibility to use models formulated in the MPS format.
-   [Rsymphony](https://cran.r-project.org/web/packages/Rsymphony/index.html) has the routine
    `Rsymphony_solve_LP()` that interfaces the SYMPHONY solver for
    mixed-integer linear programs. (SYMPHONY is part of the
    [Computational Infrastructure for Operations
    Research](http://www.coin-or.org/) (COIN-OR) project.) Package
    `lpsymphony` in Bioconductor provides a similar interface to
    SYMPHONY which is easier to install.
-   The NOMAD solver is implemented in the [crs](https://cran.r-project.org/web/packages/crs/index.html)
    package for solving mixed integer programming problems. This
    algorithm is accessible via the `snomadr()` function and is
    primarily designed for constrained optimization of black box
    functions.
-   'Clp' and 'Cbc' are open-source solvers from the COIN-OR suite.
    'Clp' solves linear programs with continuous objective variables
    and is available through [ROI.plugin.clp](https://cran.r-project.org/web/packages/ROI.plugin.clp/index.html).
    'Cbc' is a powerful mixed integer linear programming solver (based
    on 'Clp'); package 'rcbc' can be installed from:
    `r github("dirkschumacher/rcbc")` (on Github).
-   Package [highs](https://cran.r-project.org/web/packages/highs/index.html)
    is an R interface to the HiGHS solver.
    [HiGHS](https://highs.dev/) is currently among the best open-source mixed-integer linear programming solvers.
    Furthermore, it can be used to solve quadratic optimization problems (without mixed integer constraints).

#### Interfaces to Commercial Optimizers

This section surveys interfaces to commercial solvers. Typically, the
corresponding libraries have to be installed separately.

-   Package [Rcplex](https://cran.r-project.org/web/packages/Rcplex/index.html) provides an interface to the IBM
    [CPLEX
    Optimizer](https://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/).
    CPLEX provides dual/primal simplex optimizers as well as a barrier
    optimizer for solving large-scale linear and quadratic programs. It
    offers a mixed integer optimizer to solve difficult mixed integer
    programs including (possibly non-convex) MIQCP. Note that CPLEX is
    not free and you have to get a license. Academics will receive a
    free license upon request.
-   Package [Rmosek](https://cran.r-project.org/web/packages/Rmosek/index.html)
    provides an interface to the (commercial) MOSEK optimization library for large-scale
    LP, QP, and MIP problems, with emphasis on (nonlinear) conic, semidefinite, and convex tasks.
    The solver can handle SOCP and quadratically constrained programming (QPQC) tasks and offers
    to solve difficult mixed integer programs.
    (Academic licenses are available free of charge. An article on Rmosek appeared in the
    JSS special issue on Optimization with R, see below.)
-   '[Gurobi](https://www.gurobi.com/) Optimization' ships an R package with its software
    that allows for calling its solvers from R. Gurobi provides powerful solvers for
    LP, MIP, QP, MIQP, SOCP, and MISOCP models. See their website for more details.
    (Academic licenses are available on request.)

Some more commercial companies, e.g. 'LocalSolver', 'Artelys Knitro', or
'FICO Xpress Optimization', have R interfaces that are installed while the
software gets installed. Trial licenses are available, see the corresponding
websites for more information.


### [Combinatorial Optimization]{#discrete}

-   Package [adagio](https://cran.r-project.org/web/packages/adagio/index.html) provides R functions for single
    and multiple knapsack and bin packing problems, solves subset sum, maximal sum subarray, empty rectangle and
    set cover problems, and finds Hamiltonian paths in graphs.
-   In package [clue](https://cran.r-project.org/web/packages/clue/index.html) `solve_LSAP()` enables the user
    to solve the linear sum assignment problem (LSAP) using an efficient C implementation of the Hungarian algorithm.
    And function `LAPJV()` from package [TreeDist](https://cran.r-project.org/web/packages/TreeDist/index.html)
    implements the Jonker-Volgenant algorithm to solve the Linear Sum Assignment Problem (LSAP) even faster.
-   [FLSSS](https://cran.r-project.org/web/packages/FLSSS/index.html) provides multi-threaded solvers for
    fixed-size single and multi-dimensional subset sum problems with
    optional constraints on target sum and element range, fixed-size
    single and multi-dimensional knapsack problems, binary knapsack
    problems and generalized assignment problems via exact algorithms or
    metaheuristics.
-   Package [qap](https://cran.r-project.org/web/packages/qap/index.html) solves Quadratic Assignment
    Problems (QAP) applying a simulated annealing heuristics (other
    approaches will follow).
-   [igraph](https://cran.r-project.org/web/packages/igraph/index.html), a package for graph and network
    analysis, uses the very fast igraph C library. It can be used to
    calculate shortest paths, maximal network flows, minimum spanning
    trees, etc.
-   [mknapsack](https://cran.r-project.org/web/packages/mknapsack/index.html) solves multiple knapsack problems,
    based on LP solvers such as 'lpSolve' or 'CBC'; will assign
    items to knapsacks in a way that the value of the top knapsacks is
    as large as possible.
-   Package 'knapsack' (see R-Forge project
    `r rforge("optimist")`) provides routines from the book
    `Knapsack Problems' by Martello and Toth. There are functions for
    (multiple) knapsack, subset sum, and binpacking problems. (Use of
    Fortran codes is restricted to personal research and academic purposes only.)
-   [nilde](https://cran.r-project.org/web/packages/nilde/index.html) provides routines for enumerating all
    integer solutions of linear Diophantine equations, resp. all
    solutions of knapsack, subset sum, and additive partitioning
    problems (based on a generating functions approach).
-   [matchingR](https://cran.r-project.org/web/packages/matchingR/index.html) implements
    the Gale-Shapley algorithm for stable marriage and the college admissions
    problems, the stable roommates, and the house-allocation problems.
-   Package [optmatch](https://cran.r-project.org/web/packages/optmatch/index.html) provides routines for solving
    matching problems by translating them into minimum-cost flow
    problems and then solved optimally by the RELAX-IV codes of Bertsekas
    and Tseng (free for research).
-   Package [TSP](https://cran.r-project.org/web/packages/TSP/index.html) provides basic infrastructure for
    handling and solving the traveling salesperson problem (TSP). The
    main routine `solve_TSP()` solves the TSP through several
    heuristics. In addition, it provides an interface to the [Concorde
    TSP Solver](http://www.tsp.gatech.edu/concorde/index.html), which
    has to be downloaded separately.
-   [rminizinc](https://cran.r-project.org/web/packages/rminizinc/index.html) provides an interface to the open-source constraint
    modeling language and system [MiniZinc](https://www.minizinc.org/) 
    (to be downloaded separately). R users can apply the package to solve
    combinatorial optimization problems by modifying existing 'MiniZinc'
    models, and also by creating their own models.


### [Multi Objective Optimization]{#multiobj}

-   Function `caRamel` in package [caRamel](https://cran.r-project.org/web/packages/caRamel/index.html) is a
    multi-objective optimizer, applying a combination of the
    multi-objective evolutionary annealing-simplex (MEAS) method and the
    non-dominated sorting genetic algorithm (NGSA-II); it was initially
    developed for the calibration of hydrological models.
-   Multi-criteria optimization problems can be solved using package
    [mco](https://cran.r-project.org/web/packages/mco/index.html) which implements genetic algorithms.
-   [GPareto](https://cran.r-project.org/web/packages/GPareto/index.html) provides multi-objective optimization
    algorithms for expensive black-box functions and uncertainty
    quantification methods.
-   The [rmoo](https://cran.r-project.org/web/packages/rmoo/index.html) package is a framework for multi- and
    many-objective optimization, allowing to work with the representation of
    real numbers, permutations, and binaries, offering a high range of configurations.


### Miscellaneous

-   The data cloning algorithm is a global optimization approach and a
    variant of simulated annealing which has been implemented in package
    [dclone](https://cran.r-project.org/web/packages/dclone/index.html). The package provides low level
    functions for implementing maximum likelihood estimating procedures
    for complex models.
-   The [irace](https://cran.r-project.org/web/packages/irace/index.html) package implements automatic
    configuration procedures for optimizing the parameters of other
    optimization algorithms, that is (offline) tuning their parameters
    by finding the most appropriate settings given a set of optimization
    problems.
-   Package [kofnGA](https://cran.r-project.org/web/packages/kofnGA/index.html) uses a genetic algorithm to
    choose a subset of a fixed size k from the integers 1:n, such that a
    user-supplied objective function is minimized at that subset.
-   [copulaedas](https://cran.r-project.org/web/packages/copulaedas/index.html) provides a platform where
    'estimation of distribution algorithms' (EDA) based on copulas can
    be implemented and studied; the package offers various EDAs, and
    newly developed EDAs can be integrated by extending an S4 class.
-   [tabuSearch](https://cran.r-project.org/web/packages/tabuSearch/index.html) implements a tabu search algorithm
    for optimizing binary strings, maximizing a user-defined target
    function, and returns the best (i.e. maximizing) binary
    configuration found.
-   Besides functionality for solving general isotone regression
    problems, package [isotone](https://cran.r-project.org/web/packages/isotone/index.html) provides a framework
    of active set methods for isotone optimization problems with
    arbitrary order restrictions.
-   [mlrMBO](https://cran.r-project.org/web/packages/mlrMBO/index.html) is a flexible and comprehensive R
    toolbox for model-based optimization ('MBO'), also known as
    Bayesian optimization. And
    [rBayesianOptimization](https://cran.r-project.org/web/packages/rBayesianOptimization/index.html) is an implementation of
    Bayesian global optimization with Gaussian Processes, for parameter
    tuning and optimization of hyperparameters.
-   The Sequential Parameter Optimization Toolbox
    [SPOT](https://cran.r-project.org/web/packages/SPOT/index.html) provides a set of tools for model-based
    optimization and tuning of algorithms. It includes surrogate models and the design of experiment approaches.
-   The [desirability](https://cran.r-project.org/web/packages/desirability/index.html) package contains S3 classes
    for multivariate optimization using the desirability function
    approach of Harrington (1965).
-   Package [sna](https://cran.r-project.org/web/packages/sna/index.html) contains the function `lab.optimize()`
    which is the front-end to a set of heuristic routines for optimizing some bivariate graph statistics.
-   [maxLik](https://cran.r-project.org/web/packages/maxLik/index.html) adds a likelihood-specific layer on top
    of a number of maximization routines like Brendt-Hall-Hall-Hausman
    (BHHH) and Newton-Raphson among others. It includes a summary and
    print methods that extract the standard errors based on the Hessian
    matrix and allows easy swapping of maximization algorithms.


### References

-   JSS Article: [ROI: An Extensible R Optimization Infrastructure (Theußl, Schwendinger, Hornik)](https://www.jstatsoft.org/article/view/v094i15)
-   JSS Article: [CVXR: An R Package for Disciplined Convex Optimization (Fu, Narasimhan, Boyd)](https://www.jstatsoft.org/article/view/v094i14)
-   JSS Special Issue: [Numerical Optimization in R: Beyond optim (Ed.: R. Varadhan)](https://www.jstatsoft.org/v60)
-   Textbook: [Nonlinear Parameter Optimization Using R Tools (J.C. Nash)](https://www.wiley.com/en-us/Nonlinear+Parameter+Optimization+Using+R+Tools-p-9781118569283)
-   Textbook: [Modern Optimization With R (P. Cortez)](https://link.springer.com/book/10.1007/978-3-030-72819-9)
-   Textbook: [Numerical Optimization (Nocedal, Wright)](https://link.springer.com/book/10.1007/978-0-387-40065-5)


### Core packages




### Related links

-   [COIN-OR Projects](https://www.coin-or.org/)
-   [NEOS Optimization Guide](https://www.neos-guide.org/Optimization-Guide)
-   [Decision Tree for Optimization Software](http://plato.asu.edu/sub/pns.html)


### Other resources

-   Cheatsheet: [Base R Optim Cheatsheet](https://github.com/hwborchers/CheatSheets/blob/main/Base%20R%20Optim%20Cheatsheet.pdf)
-   Tutorial: [CVXR Tutorial](https://github.com/bnaras/cvxr_tutorial) and [Examples](https://cvxr.rbind.io/examples/)
-   Manual: [NLopt Manual (S. Johnson)](https://nlopt.readthedocs.io/en/latest/NLopt_manual/)

