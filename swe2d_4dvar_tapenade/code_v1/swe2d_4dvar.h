#ifndef SWE2D_4DVAR_H
#define SWE2D_4DVAR_H
/* Arrays: contiguous float64 [component=3][ny][nx]. Checkpoints include t=0.
 * All functions return 0 on success; 1=bad arguments, 2=allocation failure,
 * 3=nonfinite/dry state, 4=CFL limit exceeded. No global mutable state.
 * Diagnostic array: minimum stage depth, maximum 2-D Courant number.
 * Observations are height at obs_idx, at stride, 2*stride, ..., nsteps.
 * swe_evaluate returns ONLY the observation cost and its initial-state gradient.
 * Set gradient=NULL for a forward-only cost evaluation.
 */
int swe_forward(const double *initial, int nx, int ny, double lx, double ly,
                double dt, int nsteps, int stride, double *checkpoints, double *diagnostics);
int swe_evaluate(const double *initial, int nx, int ny, double lx, double ly,
                 double dt, int nsteps, int stride, int nobs, const int *obs_idx,
                 const double *observations, double obs_std, double *cost,
                 double *gradient, double *diagnostics);
#endif
