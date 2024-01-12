% FORCESNLPsolverDynObstacle1 - a fast solver generated by FORCESPRO v6.2.0
%
%   OUTPUT = FORCESNLPsolverDynObstacle1(PARAMS) solves a multistage problem
%   subject to the parameters supplied in the following struct:
%       PARAMS.xinit - matrix of size [6x1]
%       PARAMS.x0 - matrix of size [80x1]
%       PARAMS.all_parameters - matrix of size [208x1]
%
%   OUTPUT returns the values of the last iteration of the solver where
%       OUTPUT.x1 - column vector of length 10
%       OUTPUT.x2 - column vector of length 10
%       OUTPUT.x3 - column vector of length 10
%       OUTPUT.x4 - column vector of length 10
%       OUTPUT.x5 - column vector of length 10
%       OUTPUT.x6 - column vector of length 10
%       OUTPUT.x7 - column vector of length 10
%       OUTPUT.x8 - column vector of length 10
%
%   [OUTPUT, EXITFLAG] = FORCESNLPsolverDynObstacle1(PARAMS) returns additionally
%   the integer EXITFLAG indicating the state of the solution with 
%       1 - OPTIMAL solution has been found (subject to desired accuracy)
%       0 - Timeout - maximum number of iterations reached
%      -6 - NaN or INF occured during evaluation of functions and derivatives. Please check your initial guess.
%      -7 - Method could not progress. Problem may be infeasible. Run FORCESdiagnostics on your problem to check for most common errors in the formulation.
%     -98 - Thread error
%     -99 - Locking mechanism error
%    -100 - License error
%    -101 - Insufficient number of internal memory instances
%    -102 - Number of threads larger than specified
%
%   [OUTPUT, EXITFLAG, INFO] = FORCESNLPsolverDynObstacle1(PARAMS) returns 
%   additional information about the last iterate:
%       INFO.it - scalar: iteration number
%       INFO.it2opt - scalar: number of iterations needed to optimality (branch-and-bound)
%       INFO.res_eq - scalar: inf-norm of equality constraint residuals
%       INFO.res_ineq - scalar: inf-norm of inequality constraint residuals
%       INFO.rsnorm - scalar: norm of stationarity condition
%       INFO.rcompnorm - scalar: max of all complementarity violations
%       INFO.pobj - scalar: primal objective
%       INFO.dobj - scalar: dual objective
%       INFO.dgap - scalar: duality gap := pobj - dobj
%       INFO.rdgap - scalar: relative duality gap := |dgap / pobj |
%       INFO.mu - scalar: duality measure
%       INFO.mu_aff - scalar: duality measure (after affine step)
%       INFO.sigma - scalar: centering parameter
%       INFO.lsit_aff - scalar: number of backtracking line search steps (affine direction)
%       INFO.lsit_cc - scalar: number of backtracking line search steps (combined direction)
%       INFO.step_aff - scalar: step size (affine direction)
%       INFO.step_cc - scalar: step size (combined direction)
%       INFO.solvetime - scalar: total solve time
%       INFO.fevalstime - scalar: time spent in function evaluations
%       INFO.solver_id - column vector of length 8: solver ID of FORCESPRO solver
%
% See also COPYING