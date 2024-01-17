/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) FrankaFORCESNLPsolver_dyn_obsts_model_ ## ID
#endif

#include <math.h> 
#include "FrankaFORCESNLPsolver_dyn_obsts_model.h"

#ifndef casadi_real
#define casadi_real FrankaFORCESNLPsolver_dyn_obsts_float
#endif

#ifndef casadi_int
#define casadi_int solver_int32_default
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_f2 CASADI_PREFIX(f2)
#define casadi_f3 CASADI_PREFIX(f3)
#define casadi_f4 CASADI_PREFIX(f4)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_sign CASADI_PREFIX(sign)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#if 0
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

casadi_real casadi_sign(casadi_real x) { return x<0 ? -1 : x>0 ? 1 : x;}

static const casadi_int casadi_s0[11] = {7, 1, 0, 7, 0, 1, 2, 3, 4, 5, 6};
static const casadi_int casadi_s1[22] = {18, 1, 0, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[15] = {1, 7, 0, 1, 2, 2, 3, 4, 5, 5, 0, 0, 0, 0, 0};
static const casadi_int casadi_s4[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s5[15] = {3, 7, 0, 1, 2, 2, 2, 3, 4, 5, 0, 1, 0, 1, 2};
static const casadi_int casadi_s6[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s7[16] = {2, 7, 0, 0, 0, 0, 2, 4, 6, 6, 0, 1, 0, 1, 0, 1};
static const casadi_int casadi_s8[13] = {1, 7, 0, 1, 2, 2, 3, 3, 3, 3, 0, 0, 0};

/* FrankaFORCESNLPsolver_dyn_obsts_objective_0:(i0[7],i1[18])->(o0,o1[1x7,5nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6, a7, a8;
  a0=arg[0]? arg[0][4] : 0;
  a1=arg[1]? arg[1][0] : 0;
  a0=(a0-a1);
  a1=casadi_sq(a0);
  a2=arg[0]? arg[0][5] : 0;
  a3=arg[1]? arg[1][1] : 0;
  a2=(a2-a3);
  a3=casadi_sq(a2);
  a1=(a1+a3);
  a3=10.;
  a4=arg[0]? arg[0][0] : 0;
  a5=casadi_sq(a4);
  a6=arg[0]? arg[0][1] : 0;
  a7=casadi_sq(a6);
  a5=(a5+a7);
  a5=(a3*a5);
  a1=(a1+a5);
  a5=100.;
  a7=arg[0]? arg[0][3] : 0;
  a8=casadi_sq(a7);
  a8=(a5*a8);
  a1=(a1+a8);
  if (res[0]!=0) res[0][0]=a1;
  a4=(a4+a4);
  a4=(a3*a4);
  if (res[1]!=0) res[1][0]=a4;
  a6=(a6+a6);
  a3=(a3*a6);
  if (res[1]!=0) res[1][1]=a3;
  a7=(a7+a7);
  a5=(a5*a7);
  if (res[1]!=0) res[1][2]=a5;
  a0=(a0+a0);
  if (res[1]!=0) res[1][3]=a0;
  a2=(a2+a2);
  if (res[1]!=0) res[1][4]=a2;
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_0_alloc_mem(void) {
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_0_init_mem(int mem) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_objective_0_free_mem(int mem) {
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_0_checkout(void) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_objective_0_release(int mem) {
}

void FrankaFORCESNLPsolver_dyn_obsts_objective_0_incref(void) {
}

void FrankaFORCESNLPsolver_dyn_obsts_objective_0_decref(void) {
}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_objective_0_n_in(void) { return 2;}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_objective_0_n_out(void) { return 2;}

casadi_real FrankaFORCESNLPsolver_dyn_obsts_objective_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_objective_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_objective_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_objective_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_objective_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    default: return 0;
  }
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FrankaFORCESNLPsolver_dyn_obsts_dynamics_0:(i0[7],i1[18])->(o0[3],o1[3x7,5nz]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1;
  a0=arg[0]? arg[0][4] : 0;
  a1=arg[0]? arg[0][0] : 0;
  a0=(a0+a1);
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][5] : 0;
  a1=arg[0]? arg[0][1] : 0;
  a0=(a0+a1);
  if (res[0]!=0) res[0][1]=a0;
  a0=arg[0]? arg[0][6] : 0;
  if (res[0]!=0) res[0][2]=a0;
  a0=1.;
  if (res[1]!=0) res[1][0]=a0;
  if (res[1]!=0) res[1][1]=a0;
  if (res[1]!=0) res[1][2]=a0;
  if (res[1]!=0) res[1][3]=a0;
  if (res[1]!=0) res[1][4]=a0;
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_dynamics_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f1(arg, res, iw, w, mem);
}

int FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_alloc_mem(void) {
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_init_mem(int mem) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_free_mem(int mem) {
}

int FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_checkout(void) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_release(int mem) {
}

void FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_incref(void) {
}

void FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_decref(void) {
}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_n_in(void) { return 2;}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_n_out(void) { return 2;}

casadi_real FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    default: return 0;
  }
}

int FrankaFORCESNLPsolver_dyn_obsts_dynamics_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FrankaFORCESNLPsolver_dyn_obsts_inequalities_0:(i0[7],i1[18])->(o0[2],o1[2x7,6nz]) */
static int casadi_f2(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][4] : 0;
  a1=arg[1]? arg[1][6] : 0;
  a1=(a0-a1);
  a2=fabs(a1);
  a3=arg[1]? arg[1][9] : 0;
  a4=1.4999999999999999e-02;
  a3=(a3+a4);
  a2=(a2/a3);
  a4=6.;
  a5=(a4*a2);
  a5=exp(a5);
  a6=(a2*a5);
  a7=arg[0]? arg[0][5] : 0;
  a8=arg[1]? arg[1][7] : 0;
  a8=(a7-a8);
  a9=fabs(a8);
  a10=arg[1]? arg[1][10] : 0;
  a11=3.7999999999999999e-02;
  a10=(a10+a11);
  a9=(a9/a10);
  a11=(a4*a9);
  a11=exp(a11);
  a12=(a9*a11);
  a6=(a6+a12);
  a12=(a4*a2);
  a12=exp(a12);
  a13=(a4*a9);
  a13=exp(a13);
  a14=(a12+a13);
  a6=(a6/a14);
  a15=arg[0]? arg[0][3] : 0;
  a16=(a6+a15);
  if (res[0]!=0) res[0][0]=a16;
  a16=arg[1]? arg[1][12] : 0;
  a0=(a0-a16);
  a16=casadi_sq(a0);
  a17=arg[1]? arg[1][13] : 0;
  a7=(a7-a17);
  a17=casadi_sq(a7);
  a16=(a16+a17);
  a16=sqrt(a16);
  a15=(a16+a15);
  if (res[0]!=0) res[0][1]=a15;
  a15=1.;
  if (res[1]!=0) res[1][0]=a15;
  if (res[1]!=0) res[1][1]=a15;
  a1=casadi_sign(a1);
  a1=(a1/a3);
  a3=(a5*a1);
  a15=(a4*a1);
  a5=(a5*a15);
  a2=(a2*a5);
  a3=(a3+a2);
  a3=(a3/a14);
  a6=(a6/a14);
  a1=(a4*a1);
  a12=(a12*a1);
  a12=(a6*a12);
  a3=(a3-a12);
  if (res[1]!=0) res[1][2]=a3;
  a0=(a0/a16);
  if (res[1]!=0) res[1][3]=a0;
  a8=casadi_sign(a8);
  a8=(a8/a10);
  a10=(a11*a8);
  a0=(a4*a8);
  a11=(a11*a0);
  a9=(a9*a11);
  a10=(a10+a9);
  a10=(a10/a14);
  a4=(a4*a8);
  a13=(a13*a4);
  a6=(a6*a13);
  a10=(a10-a6);
  if (res[1]!=0) res[1][4]=a10;
  a7=(a7/a16);
  if (res[1]!=0) res[1][5]=a7;
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f2(arg, res, iw, w, mem);
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_alloc_mem(void) {
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_init_mem(int mem) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_free_mem(int mem) {
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_checkout(void) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_release(int mem) {
}

void FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_incref(void) {
}

void FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_decref(void) {
}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_n_in(void) { return 2;}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_n_out(void) { return 2;}

casadi_real FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s6;
    case 1: return casadi_s7;
    default: return 0;
  }
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FrankaFORCESNLPsolver_dyn_obsts_objective_1:(i0[7],i1[18])->(o0,o1[1x7,3nz]) */
static int casadi_f3(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4, a5, a6;
  a0=10.;
  a1=arg[0]? arg[0][0] : 0;
  a2=casadi_sq(a1);
  a3=arg[0]? arg[0][1] : 0;
  a4=casadi_sq(a3);
  a2=(a2+a4);
  a2=(a0*a2);
  a4=100.;
  a5=arg[0]? arg[0][3] : 0;
  a6=casadi_sq(a5);
  a6=(a4*a6);
  a2=(a2+a6);
  if (res[0]!=0) res[0][0]=a2;
  a1=(a1+a1);
  a1=(a0*a1);
  if (res[1]!=0) res[1][0]=a1;
  a3=(a3+a3);
  a0=(a0*a3);
  if (res[1]!=0) res[1][1]=a0;
  a5=(a5+a5);
  a4=(a4*a5);
  if (res[1]!=0) res[1][2]=a4;
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f3(arg, res, iw, w, mem);
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_1_alloc_mem(void) {
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_1_init_mem(int mem) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_objective_1_free_mem(int mem) {
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_1_checkout(void) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_objective_1_release(int mem) {
}

void FrankaFORCESNLPsolver_dyn_obsts_objective_1_incref(void) {
}

void FrankaFORCESNLPsolver_dyn_obsts_objective_1_decref(void) {
}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_objective_1_n_in(void) { return 2;}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_objective_1_n_out(void) { return 2;}

casadi_real FrankaFORCESNLPsolver_dyn_obsts_objective_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_objective_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_objective_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_objective_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_objective_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s8;
    default: return 0;
  }
}

int FrankaFORCESNLPsolver_dyn_obsts_objective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

/* FrankaFORCESNLPsolver_dyn_obsts_inequalities_1:(i0[7],i1[18])->(o0[2],o1[2x7,6nz]) */
static int casadi_f4(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a2, a3, a4, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][4] : 0;
  a1=arg[1]? arg[1][6] : 0;
  a1=(a0-a1);
  a2=fabs(a1);
  a3=arg[1]? arg[1][9] : 0;
  a4=1.4999999999999999e-02;
  a3=(a3+a4);
  a2=(a2/a3);
  a4=6.;
  a5=(a4*a2);
  a5=exp(a5);
  a6=(a2*a5);
  a7=arg[0]? arg[0][5] : 0;
  a8=arg[1]? arg[1][7] : 0;
  a8=(a7-a8);
  a9=fabs(a8);
  a10=arg[1]? arg[1][10] : 0;
  a11=3.7999999999999999e-02;
  a10=(a10+a11);
  a9=(a9/a10);
  a11=(a4*a9);
  a11=exp(a11);
  a12=(a9*a11);
  a6=(a6+a12);
  a12=(a4*a2);
  a12=exp(a12);
  a13=(a4*a9);
  a13=exp(a13);
  a14=(a12+a13);
  a6=(a6/a14);
  a15=arg[0]? arg[0][3] : 0;
  a16=(a6+a15);
  if (res[0]!=0) res[0][0]=a16;
  a16=arg[1]? arg[1][12] : 0;
  a0=(a0-a16);
  a16=casadi_sq(a0);
  a17=arg[1]? arg[1][13] : 0;
  a7=(a7-a17);
  a17=casadi_sq(a7);
  a16=(a16+a17);
  a16=sqrt(a16);
  a15=(a16+a15);
  if (res[0]!=0) res[0][1]=a15;
  a15=1.;
  if (res[1]!=0) res[1][0]=a15;
  if (res[1]!=0) res[1][1]=a15;
  a1=casadi_sign(a1);
  a1=(a1/a3);
  a3=(a5*a1);
  a15=(a4*a1);
  a5=(a5*a15);
  a2=(a2*a5);
  a3=(a3+a2);
  a3=(a3/a14);
  a6=(a6/a14);
  a1=(a4*a1);
  a12=(a12*a1);
  a12=(a6*a12);
  a3=(a3-a12);
  if (res[1]!=0) res[1][2]=a3;
  a0=(a0/a16);
  if (res[1]!=0) res[1][3]=a0;
  a8=casadi_sign(a8);
  a8=(a8/a10);
  a10=(a11*a8);
  a0=(a4*a8);
  a11=(a11*a0);
  a9=(a9*a11);
  a10=(a10+a9);
  a10=(a10/a14);
  a4=(a4*a8);
  a13=(a13*a4);
  a6=(a6*a13);
  a10=(a10-a6);
  if (res[1]!=0) res[1][4]=a10;
  a7=(a7/a16);
  if (res[1]!=0) res[1][5]=a7;
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f4(arg, res, iw, w, mem);
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_alloc_mem(void) {
  return 0;
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_init_mem(int mem) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_free_mem(int mem) {
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_checkout(void) {
  return 0;
}

void FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_release(int mem) {
}

void FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_incref(void) {
}

void FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_decref(void) {
}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_n_in(void) { return 2;}

casadi_int FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_n_out(void) { return 2;}

casadi_real FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

const char* FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

const casadi_int* FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s6;
    case 1: return casadi_s7;
    default: return 0;
  }
}

int FrankaFORCESNLPsolver_dyn_obsts_inequalities_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 2;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif