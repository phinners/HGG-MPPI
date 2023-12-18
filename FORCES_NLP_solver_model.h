/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef casadi_real
#define casadi_real FORCES_NLP_solver_float
#endif

#ifndef casadi_int
#define casadi_int solver_int32_default
#endif

int FORCES_NLP_solver_objective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int FORCES_NLP_solver_objective_0_alloc_mem(void);
int FORCES_NLP_solver_objective_0_init_mem(int mem);
void FORCES_NLP_solver_objective_0_free_mem(int mem);
int FORCES_NLP_solver_objective_0_checkout(void);
void FORCES_NLP_solver_objective_0_release(int mem);
void FORCES_NLP_solver_objective_0_incref(void);
void FORCES_NLP_solver_objective_0_decref(void);
casadi_int FORCES_NLP_solver_objective_0_n_out(void);
casadi_int FORCES_NLP_solver_objective_0_n_in(void);
casadi_real FORCES_NLP_solver_objective_0_default_in(casadi_int i);
const char* FORCES_NLP_solver_objective_0_name_in(casadi_int i);
const char* FORCES_NLP_solver_objective_0_name_out(casadi_int i);
const casadi_int* FORCES_NLP_solver_objective_0_sparsity_in(casadi_int i);
const casadi_int* FORCES_NLP_solver_objective_0_sparsity_out(casadi_int i);
int FORCES_NLP_solver_objective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int FORCES_NLP_solver_dobjective_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int FORCES_NLP_solver_dobjective_0_alloc_mem(void);
int FORCES_NLP_solver_dobjective_0_init_mem(int mem);
void FORCES_NLP_solver_dobjective_0_free_mem(int mem);
int FORCES_NLP_solver_dobjective_0_checkout(void);
void FORCES_NLP_solver_dobjective_0_release(int mem);
void FORCES_NLP_solver_dobjective_0_incref(void);
void FORCES_NLP_solver_dobjective_0_decref(void);
casadi_int FORCES_NLP_solver_dobjective_0_n_out(void);
casadi_int FORCES_NLP_solver_dobjective_0_n_in(void);
casadi_real FORCES_NLP_solver_dobjective_0_default_in(casadi_int i);
const char* FORCES_NLP_solver_dobjective_0_name_in(casadi_int i);
const char* FORCES_NLP_solver_dobjective_0_name_out(casadi_int i);
const casadi_int* FORCES_NLP_solver_dobjective_0_sparsity_in(casadi_int i);
const casadi_int* FORCES_NLP_solver_dobjective_0_sparsity_out(casadi_int i);
int FORCES_NLP_solver_dobjective_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int FORCES_NLP_solver_hessian_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int FORCES_NLP_solver_hessian_0_alloc_mem(void);
int FORCES_NLP_solver_hessian_0_init_mem(int mem);
void FORCES_NLP_solver_hessian_0_free_mem(int mem);
int FORCES_NLP_solver_hessian_0_checkout(void);
void FORCES_NLP_solver_hessian_0_release(int mem);
void FORCES_NLP_solver_hessian_0_incref(void);
void FORCES_NLP_solver_hessian_0_decref(void);
casadi_int FORCES_NLP_solver_hessian_0_n_out(void);
casadi_int FORCES_NLP_solver_hessian_0_n_in(void);
casadi_real FORCES_NLP_solver_hessian_0_default_in(casadi_int i);
const char* FORCES_NLP_solver_hessian_0_name_in(casadi_int i);
const char* FORCES_NLP_solver_hessian_0_name_out(casadi_int i);
const casadi_int* FORCES_NLP_solver_hessian_0_sparsity_in(casadi_int i);
const casadi_int* FORCES_NLP_solver_hessian_0_sparsity_out(casadi_int i);
int FORCES_NLP_solver_hessian_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int FORCES_NLP_solver_cdyn_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int FORCES_NLP_solver_cdyn_0_alloc_mem(void);
int FORCES_NLP_solver_cdyn_0_init_mem(int mem);
void FORCES_NLP_solver_cdyn_0_free_mem(int mem);
int FORCES_NLP_solver_cdyn_0_checkout(void);
void FORCES_NLP_solver_cdyn_0_release(int mem);
void FORCES_NLP_solver_cdyn_0_incref(void);
void FORCES_NLP_solver_cdyn_0_decref(void);
casadi_int FORCES_NLP_solver_cdyn_0_n_out(void);
casadi_int FORCES_NLP_solver_cdyn_0_n_in(void);
casadi_real FORCES_NLP_solver_cdyn_0_default_in(casadi_int i);
const char* FORCES_NLP_solver_cdyn_0_name_in(casadi_int i);
const char* FORCES_NLP_solver_cdyn_0_name_out(casadi_int i);
const casadi_int* FORCES_NLP_solver_cdyn_0_sparsity_in(casadi_int i);
const casadi_int* FORCES_NLP_solver_cdyn_0_sparsity_out(casadi_int i);
int FORCES_NLP_solver_cdyn_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int FORCES_NLP_solver_cdyn_0rd_0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int FORCES_NLP_solver_cdyn_0rd_0_alloc_mem(void);
int FORCES_NLP_solver_cdyn_0rd_0_init_mem(int mem);
void FORCES_NLP_solver_cdyn_0rd_0_free_mem(int mem);
int FORCES_NLP_solver_cdyn_0rd_0_checkout(void);
void FORCES_NLP_solver_cdyn_0rd_0_release(int mem);
void FORCES_NLP_solver_cdyn_0rd_0_incref(void);
void FORCES_NLP_solver_cdyn_0rd_0_decref(void);
casadi_int FORCES_NLP_solver_cdyn_0rd_0_n_out(void);
casadi_int FORCES_NLP_solver_cdyn_0rd_0_n_in(void);
casadi_real FORCES_NLP_solver_cdyn_0rd_0_default_in(casadi_int i);
const char* FORCES_NLP_solver_cdyn_0rd_0_name_in(casadi_int i);
const char* FORCES_NLP_solver_cdyn_0rd_0_name_out(casadi_int i);
const casadi_int* FORCES_NLP_solver_cdyn_0rd_0_sparsity_in(casadi_int i);
const casadi_int* FORCES_NLP_solver_cdyn_0rd_0_sparsity_out(casadi_int i);
int FORCES_NLP_solver_cdyn_0rd_0_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int FORCES_NLP_solver_objective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int FORCES_NLP_solver_objective_1_alloc_mem(void);
int FORCES_NLP_solver_objective_1_init_mem(int mem);
void FORCES_NLP_solver_objective_1_free_mem(int mem);
int FORCES_NLP_solver_objective_1_checkout(void);
void FORCES_NLP_solver_objective_1_release(int mem);
void FORCES_NLP_solver_objective_1_incref(void);
void FORCES_NLP_solver_objective_1_decref(void);
casadi_int FORCES_NLP_solver_objective_1_n_out(void);
casadi_int FORCES_NLP_solver_objective_1_n_in(void);
casadi_real FORCES_NLP_solver_objective_1_default_in(casadi_int i);
const char* FORCES_NLP_solver_objective_1_name_in(casadi_int i);
const char* FORCES_NLP_solver_objective_1_name_out(casadi_int i);
const casadi_int* FORCES_NLP_solver_objective_1_sparsity_in(casadi_int i);
const casadi_int* FORCES_NLP_solver_objective_1_sparsity_out(casadi_int i);
int FORCES_NLP_solver_objective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int FORCES_NLP_solver_dobjective_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int FORCES_NLP_solver_dobjective_1_alloc_mem(void);
int FORCES_NLP_solver_dobjective_1_init_mem(int mem);
void FORCES_NLP_solver_dobjective_1_free_mem(int mem);
int FORCES_NLP_solver_dobjective_1_checkout(void);
void FORCES_NLP_solver_dobjective_1_release(int mem);
void FORCES_NLP_solver_dobjective_1_incref(void);
void FORCES_NLP_solver_dobjective_1_decref(void);
casadi_int FORCES_NLP_solver_dobjective_1_n_out(void);
casadi_int FORCES_NLP_solver_dobjective_1_n_in(void);
casadi_real FORCES_NLP_solver_dobjective_1_default_in(casadi_int i);
const char* FORCES_NLP_solver_dobjective_1_name_in(casadi_int i);
const char* FORCES_NLP_solver_dobjective_1_name_out(casadi_int i);
const casadi_int* FORCES_NLP_solver_dobjective_1_sparsity_in(casadi_int i);
const casadi_int* FORCES_NLP_solver_dobjective_1_sparsity_out(casadi_int i);
int FORCES_NLP_solver_dobjective_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
int FORCES_NLP_solver_hessian_1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
int FORCES_NLP_solver_hessian_1_alloc_mem(void);
int FORCES_NLP_solver_hessian_1_init_mem(int mem);
void FORCES_NLP_solver_hessian_1_free_mem(int mem);
int FORCES_NLP_solver_hessian_1_checkout(void);
void FORCES_NLP_solver_hessian_1_release(int mem);
void FORCES_NLP_solver_hessian_1_incref(void);
void FORCES_NLP_solver_hessian_1_decref(void);
casadi_int FORCES_NLP_solver_hessian_1_n_out(void);
casadi_int FORCES_NLP_solver_hessian_1_n_in(void);
casadi_real FORCES_NLP_solver_hessian_1_default_in(casadi_int i);
const char* FORCES_NLP_solver_hessian_1_name_in(casadi_int i);
const char* FORCES_NLP_solver_hessian_1_name_out(casadi_int i);
const casadi_int* FORCES_NLP_solver_hessian_1_sparsity_in(casadi_int i);
const casadi_int* FORCES_NLP_solver_hessian_1_sparsity_out(casadi_int i);
int FORCES_NLP_solver_hessian_1_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#ifdef __cplusplus
} /* extern "C" */
#endif
