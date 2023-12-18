/*
FORCESNLPsolverDynLiftedObstacle1 : A fast customized optimization solver.

Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.


This software is intended for simulation and testing purposes only. 
Use of this software for any commercial purpose is prohibited.

This program is distributed in the hope that it will be useful.
EMBOTECH makes NO WARRANTIES with respect to the use of the software 
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. 

EMBOTECH shall not have any liability for any damage arising from the use
of the software.

This Agreement shall exclusively be governed by and interpreted in 
accordance with the laws of Switzerland, excluding its principles
of conflict of laws. The Courts of Zurich-City shall have exclusive 
jurisdiction in case of any dispute.

*/

/* Generated by FORCESPRO v6.2.0 on Monday, December 18, 2023 at 8:09:28 AM */
#ifndef FORCESNLPsolverDynLiftedObstacle1_H
#define FORCESNLPsolverDynLiftedObstacle1_H

#ifndef SOLVER_STDIO_H
#define SOLVER_STDIO_H
#include <stdio.h>
#endif
#ifndef SOLVER_STRING_H
#define SOLVER_STRING_H
#include <string.h>
#endif


#ifndef SOLVER_STANDARD_TYPES
#define SOLVER_STANDARD_TYPES

typedef signed char solver_int8_signed;
typedef unsigned char solver_int8_unsigned;
typedef char solver_int8_default;
typedef signed short int solver_int16_signed;
typedef unsigned short int solver_int16_unsigned;
typedef short int solver_int16_default;
typedef signed int solver_int32_signed;
typedef unsigned int solver_int32_unsigned;
typedef int solver_int32_default;
typedef signed long long int solver_int64_signed;
typedef unsigned long long int solver_int64_unsigned;
typedef long long int solver_int64_default;

#endif


/* DATA TYPE ------------------------------------------------------------*/
typedef double FORCESNLPsolverDynLiftedObstacle1_float;
typedef double FORCESNLPsolverDynLiftedObstacle1_ldl_s_float;
typedef double FORCESNLPsolverDynLiftedObstacle1_ldl_r_float;
typedef double FORCESNLPsolverDynLiftedObstacle1_callback_float;

typedef double FORCESNLPsolverDynLiftedObstacle1interface_float;

/* SOLVER SETTINGS ------------------------------------------------------*/

/* MISRA-C compliance */
#ifndef MISRA_C_FORCESNLPsolverDynLiftedObstacle1
#define MISRA_C_FORCESNLPsolverDynLiftedObstacle1 (0)
#endif

/* restrict code */
#ifndef RESTRICT_CODE_FORCESNLPsolverDynLiftedObstacle1
#define RESTRICT_CODE_FORCESNLPsolverDynLiftedObstacle1 (0)
#endif

/* print level */
#ifndef SET_PRINTLEVEL_FORCESNLPsolverDynLiftedObstacle1
#define SET_PRINTLEVEL_FORCESNLPsolverDynLiftedObstacle1    (0)
#endif

/* timing */
#ifndef SET_TIMING_FORCESNLPsolverDynLiftedObstacle1
#define SET_TIMING_FORCESNLPsolverDynLiftedObstacle1    (1)
#endif

/* Numeric Warnings */
/* #define PRINTNUMERICALWARNINGS */

/* maximum number of iterations  */
#define SET_MAXIT_FORCESNLPsolverDynLiftedObstacle1			(200)	

/* scaling factor of line search (FTB rule) */
#define SET_FLS_SCALE_FORCESNLPsolverDynLiftedObstacle1		(FORCESNLPsolverDynLiftedObstacle1_float)(0.99)      

/* maximum number of supported elements in the filter */
#define MAX_FILTER_SIZE_FORCESNLPsolverDynLiftedObstacle1	(200) 

/* whether callback return values should be checked */
#define EXTFUNC_RETURN_FORCESNLPsolverDynLiftedObstacle1 (0)

/* SOLVER RETURN CODES----------------------------------------------------------*/
/* solver has converged within desired accuracy */
#define OPTIMAL_FORCESNLPsolverDynLiftedObstacle1      (1)

/* maximum number of iterations has been reached */
#define MAXITREACHED_FORCESNLPsolverDynLiftedObstacle1 (0)

/* solver has stopped due to a timeout */
#define TIMEOUT_FORCESNLPsolverDynLiftedObstacle1   (2)

/* solver stopped externally */
#define EXIT_EXTERNAL_FORCESNLPsolverDynLiftedObstacle1 (3)

/* wrong number of inequalities error */
#define INVALID_NUM_INEQ_ERROR_FORCESNLPsolverDynLiftedObstacle1  (-4)

/* factorization error */
#define FACTORIZATION_ERROR_FORCESNLPsolverDynLiftedObstacle1   (-5)

/* NaN encountered in function evaluations */
#define BADFUNCEVAL_FORCESNLPsolverDynLiftedObstacle1  (-6)

/* invalid value (<= -100) returned by external function */
#define INVALIDFUNCEXIT_FORCESNLPsolverDynLiftedObstacle1 (-200)

/* bad value returned by external function */
#define BADFUNCEXIT_FORCESNLPsolverDynLiftedObstacle1(status) (status > -100? status - 200 : INVALIDFUNCEXIT_FORCESNLPsolverDynLiftedObstacle1)

/* no progress in method possible */
#define NOPROGRESS_FORCESNLPsolverDynLiftedObstacle1   (-7)

/* regularization error */
#define REGULARIZATION_ERROR_FORCESNLPsolverDynLiftedObstacle1   (-9)

/* invalid values in parameters */
#define PARAM_VALUE_ERROR_FORCESNLPsolverDynLiftedObstacle1   (-11)

/* too small timeout given */
#define INVALID_TIMEOUT_FORCESNLPsolverDynLiftedObstacle1   (-12)

/* thread error */
#define THREAD_FAILURE_FORCESNLPsolverDynLiftedObstacle1  (-98)

/* locking mechanism error */
#define LOCK_FAILURE_FORCESNLPsolverDynLiftedObstacle1  (-99)

/* licensing error - solver not valid on this machine */
#define LICENSE_ERROR_FORCESNLPsolverDynLiftedObstacle1  (-100)

/* Insufficient number of internal memory instances.
 * Increase codeoptions.max_num_mem. */
#define MEMORY_INVALID_FORCESNLPsolverDynLiftedObstacle1 (-101)
/* Number of threads larger than specified.
 * Increase codeoptions.nlp.max_num_threads. */
#define NUMTHREADS_INVALID_FORCESNLPsolverDynLiftedObstacle1 (-102)


/* INTEGRATORS RETURN CODE ------------*/
/* Integrator ran successfully */
#define INTEGRATOR_SUCCESS (11)
/* Number of steps set by user exceeds maximum number of steps allowed */
#define INTEGRATOR_MAXSTEPS_EXCEEDED (12)


/* MEMORY STRUCT --------------------------------------------------------*/
typedef struct FORCESNLPsolverDynLiftedObstacle1_mem FORCESNLPsolverDynLiftedObstacle1_mem;
#ifdef __cplusplus
extern "C" {
#endif
/* MEMORY STRUCT --------------------------------------------------------*/
extern FORCESNLPsolverDynLiftedObstacle1_mem * FORCESNLPsolverDynLiftedObstacle1_external_mem(void * mem_ptr, solver_int32_unsigned i_mem, size_t mem_size);
extern size_t FORCESNLPsolverDynLiftedObstacle1_get_mem_size( void );
extern size_t FORCESNLPsolverDynLiftedObstacle1_get_const_size( void );
#ifdef __cplusplus
}
#endif

/* PARAMETERS -----------------------------------------------------------*/
/* fill this with data before calling the solver! */
typedef struct
{
    /* vector of size 6 */
    FORCESNLPsolverDynLiftedObstacle1_float xinit[6];

    /* vector of size 80 */
    FORCESNLPsolverDynLiftedObstacle1_float x0[80];

    /* vector of size 192 */
    FORCESNLPsolverDynLiftedObstacle1_float all_parameters[192];


} FORCESNLPsolverDynLiftedObstacle1_params;


/* OUTPUTS --------------------------------------------------------------*/
/* the desired variables are put here by the solver */
typedef struct
{
    /* column vector of length 10 */
    FORCESNLPsolverDynLiftedObstacle1_float x1[10];

    /* column vector of length 10 */
    FORCESNLPsolverDynLiftedObstacle1_float x2[10];

    /* column vector of length 10 */
    FORCESNLPsolverDynLiftedObstacle1_float x3[10];

    /* column vector of length 10 */
    FORCESNLPsolverDynLiftedObstacle1_float x4[10];

    /* column vector of length 10 */
    FORCESNLPsolverDynLiftedObstacle1_float x5[10];

    /* column vector of length 10 */
    FORCESNLPsolverDynLiftedObstacle1_float x6[10];

    /* column vector of length 10 */
    FORCESNLPsolverDynLiftedObstacle1_float x7[10];

    /* column vector of length 10 */
    FORCESNLPsolverDynLiftedObstacle1_float x8[10];


} FORCESNLPsolverDynLiftedObstacle1_output;


/* SOLVER INFO ----------------------------------------------------------*/
/* diagnostic data from last interior point step */
typedef struct
{
    /* scalar: iteration number */
    solver_int32_default it;

    /* scalar: number of iterations needed to optimality (branch-and-bound) */
    solver_int32_default it2opt;

    /* scalar: inf-norm of equality constraint residuals */
    FORCESNLPsolverDynLiftedObstacle1_float res_eq;

    /* scalar: inf-norm of inequality constraint residuals */
    FORCESNLPsolverDynLiftedObstacle1_float res_ineq;

    /* scalar: norm of stationarity condition */
    FORCESNLPsolverDynLiftedObstacle1_float rsnorm;

    /* scalar: max of all complementarity violations */
    FORCESNLPsolverDynLiftedObstacle1_float rcompnorm;

    /* scalar: primal objective */
    FORCESNLPsolverDynLiftedObstacle1_float pobj;

    /* scalar: dual objective */
    FORCESNLPsolverDynLiftedObstacle1_float dobj;

    /* scalar: duality gap := pobj - dobj */
    FORCESNLPsolverDynLiftedObstacle1_float dgap;

    /* scalar: relative duality gap := |dgap / pobj | */
    FORCESNLPsolverDynLiftedObstacle1_float rdgap;

    /* scalar: duality measure */
    FORCESNLPsolverDynLiftedObstacle1_float mu;

    /* scalar: duality measure (after affine step) */
    FORCESNLPsolverDynLiftedObstacle1_float mu_aff;

    /* scalar: centering parameter */
    FORCESNLPsolverDynLiftedObstacle1_float sigma;

    /* scalar: number of backtracking line search steps (affine direction) */
    solver_int32_default lsit_aff;

    /* scalar: number of backtracking line search steps (combined direction) */
    solver_int32_default lsit_cc;

    /* scalar: step size (affine direction) */
    FORCESNLPsolverDynLiftedObstacle1_float step_aff;

    /* scalar: step size (combined direction) */
    FORCESNLPsolverDynLiftedObstacle1_float step_cc;

    /* scalar: total solve time */
    FORCESNLPsolverDynLiftedObstacle1_float solvetime;

    /* scalar: time spent in function evaluations */
    FORCESNLPsolverDynLiftedObstacle1_float fevalstime;

    /* column vector of length 8: solver ID of FORCESPRO solver */
    solver_int32_default solver_id[8];




} FORCESNLPsolverDynLiftedObstacle1_info;







/* SOLVER FUNCTION DEFINITION -------------------------------------------*/
/* Time of Solver Generation: (UTC) Monday, December 18, 2023 8:09:29 AM */
/* User License expires on: (UTC) Monday, January 29, 2024 10:00:00 PM (approx.) (at the time of code generation) */
/* Solver Static License expires on: (UTC) Monday, January 29, 2024 10:00:00 PM (approx.) */
/* Solver Id: ab97dadd-2fef-498b-905c-7e0f865d8d18 */
/* Host Compiler Version: d76500f0 */
/* Target Compiler Version: unused */
/* examine exitflag before using the result! */
#ifdef __cplusplus
extern "C" {
#endif		

typedef solver_int32_default (*FORCESNLPsolverDynLiftedObstacle1_extfunc)(FORCESNLPsolverDynLiftedObstacle1_float* x, FORCESNLPsolverDynLiftedObstacle1_float* y, FORCESNLPsolverDynLiftedObstacle1_float* lambda, FORCESNLPsolverDynLiftedObstacle1_float* params, FORCESNLPsolverDynLiftedObstacle1_float* pobj, FORCESNLPsolverDynLiftedObstacle1_float* g, FORCESNLPsolverDynLiftedObstacle1_float* c, FORCESNLPsolverDynLiftedObstacle1_float* Jeq, FORCESNLPsolverDynLiftedObstacle1_float* h, FORCESNLPsolverDynLiftedObstacle1_float* Jineq, FORCESNLPsolverDynLiftedObstacle1_float* H, solver_int32_default stage, solver_int32_default iterations, solver_int32_default threadID);

extern solver_int32_default FORCESNLPsolverDynLiftedObstacle1_solve(FORCESNLPsolverDynLiftedObstacle1_params *params, FORCESNLPsolverDynLiftedObstacle1_output *output, FORCESNLPsolverDynLiftedObstacle1_info *info, FORCESNLPsolverDynLiftedObstacle1_mem *mem, FILE *fs, FORCESNLPsolverDynLiftedObstacle1_extfunc evalextfunctions_FORCESNLPsolverDynLiftedObstacle1);



/*Integrator declarations */
typedef const solver_int32_default* (*cDynJacXsparsity)( solver_int32_default i );
typedef const solver_int32_default* (*cDynJacUsparsity)( solver_int32_default i );
typedef solver_int32_default (*fConDynamics)( const FORCESNLPsolverDynLiftedObstacle1_callback_float** arg, FORCESNLPsolverDynLiftedObstacle1_callback_float** res, solver_int32_default* iw, FORCESNLPsolverDynLiftedObstacle1_callback_float* w, solver_int32_default mem );
typedef solver_int32_default (*fConJacStateDynamics)( const FORCESNLPsolverDynLiftedObstacle1_callback_float** arg, FORCESNLPsolverDynLiftedObstacle1_callback_float** res, solver_int32_default* iw, FORCESNLPsolverDynLiftedObstacle1_callback_float* w, solver_int32_default mem );
typedef solver_int32_default (*fConJacInputDynamics)( const FORCESNLPsolverDynLiftedObstacle1_callback_float** arg, FORCESNLPsolverDynLiftedObstacle1_callback_float** res, solver_int32_default* iw, FORCESNLPsolverDynLiftedObstacle1_callback_float* w, solver_int32_default mem );

void FORCESNLPsolverDynLiftedObstacle1_rkfour_0(const FORCESNLPsolverDynLiftedObstacle1_callback_float * const z, const FORCESNLPsolverDynLiftedObstacle1_callback_float * const p, FORCESNLPsolverDynLiftedObstacle1_float * const c, FORCESNLPsolverDynLiftedObstacle1_float * const jacc,
            fConDynamics cDyn0rd, fConDynamics cDyn, const solver_int32_default threadID);









#ifdef __cplusplus
}
#endif

#endif
