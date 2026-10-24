#ifndef SWE2D_4DVAR_H
#define SWE2D_4DVAR_H
#include <stdint.h>
/* Arrays: contiguous float64 [component=3][ny][nx]. Checkpoints include t=0.
 * All functions return 0 on success; 1=bad arguments, 2=allocation failure,
 * 3=nonfinite/dry state, 4=CFL limit exceeded, 5=nonfinite cost/gradient or AD invariant failure.
 * AD evaluations are serialized internally (Tapenade uses global tape state).
 * Diagnostic array: minimum stage depth, maximum 2-D Courant number.
 * Observations are height at obs_idx, at stride, 2*stride, ..., nsteps.
 * swe_evaluate returns ONLY the observation cost and its initial-state gradient.
 * nsteps <= 1000000. Default binomial budget: 8 snapshots.
 * Set gradient=NULL for a forward-only cost evaluation.
 */
int swe_forward(const double *initial, int nx, int ny, double lx, double ly,
                double dt, int nsteps, int stride, double *checkpoints, double *diagnostics);
int swe_evaluate(const double *initial, int nx, int ny, double lx, double ly,
                 double dt, int nsteps, int stride, int nobs, const int *obs_idx,
                 const double *observations, double obs_std, double *cost,
                 double *gradient, double *diagnostics);
/* Explicit binomial snapshot budget, 1..90. */
int swe_evaluate_binomial(const double *initial,int nx,int ny,double lx,double ly,
                 double dt,int nsteps,int stride,int nobs,const int *obs_idx,
                 const double *observations,double obs_std,int checkpoint_budget,
                 double *cost,double *gradient,double *diagnostics);
#endif
