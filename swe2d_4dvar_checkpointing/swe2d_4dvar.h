#ifndef SWE2D_4DVAR_H
#define SWE2D_4DVAR_H
#include <stdint.h>
#define SWE_DEFAULT_CHECKPOINT_BUDGET 8
#define SWE_MAX_CHECKPOINT_BUDGET 90
#define SWE_MAX_STEPS 1000000

/* Counts for one evaluation, including its initial forward pass.
 * State storage excludes caller-owned inputs/outputs and scheduler integers.
 * Cost-only evaluations use four work states and no checkpoint slots. */
typedef struct {
  uint64_t checkpoint_slots;
  uint64_t state_storage_bytes;
  uint64_t forward_steps;
  uint64_t recomputed_steps;
  uint64_t stage_recomputations;
  uint64_t reverse_steps;
  uint64_t checkpoint_saves;
  uint64_t checkpoint_restores;
} SweCheckpointStats;
/* Arrays: contiguous float64 [component=3][ny][nx]. Checkpoints include t=0.
 * All functions return 0 on success; 1=bad arguments, 2=allocation failure,
 * 3=nonfinite/dry state, 4=CFL limit exceeded, 5=nonfinite cost/gradient
 * or checkpoint invariant failure. No global mutable state.
 * Diagnostic array: minimum stage depth, maximum 2-D Courant number.
 * Observations are height at obs_idx, at stride, 2*stride, ..., nsteps.
 * swe_evaluate returns ONLY the observation cost and its initial-state gradient.
 * Set gradient=NULL for a forward-only cost evaluation. nsteps <= SWE_MAX_STEPS.
 * swe_evaluate uses SWE_DEFAULT_CHECKPOINT_BUDGET binomial snapshot slots.
 */
int swe_forward(const double *initial, int nx, int ny, double lx, double ly,
                double dt, int nsteps, int stride, double *checkpoints, double *diagnostics);
int swe_evaluate(const double *initial, int nx, int ny, double lx, double ly,
                 double dt, int nsteps, int stride, int nobs, const int *obs_idx,
                 const double *observations, double obs_std, double *cost,
                 double *gradient, double *diagnostics);
/* Demo-derived binomial schedule, independent of observation stride.
 * budget must be 1..90; excess slots are capped at max(1,nsteps-1).
 * Each snapshot holds one full state. Six work states suffice for reverse.
 * stats may be NULL; it is reset on entry and may be partial on failure. */
int swe_evaluate_binomial(const double *initial, int nx, int ny, double lx, double ly,
                 double dt, int nsteps, int stride, int nobs, const int *obs_idx,
                 const double *observations, double obs_std, int checkpoint_budget,
                 double *cost, double *gradient, double *diagnostics,
                 SweCheckpointStats *stats);
#endif
