#ifndef __solver_h__
#define __solver_h__

typedef void (*op_t)(int, double*, double*); //define type for linear operator

double dotprod(int n, double *a, double *b);

//linear solver using conjugate gradient method
void solve_cg(int n, double *x, double *b, op_t Aop, int niter, double tol, int verb);

//linear solver using conjugate gradient method
void solve_pcg(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol, int verb);


#endif
