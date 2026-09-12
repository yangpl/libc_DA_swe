/* main.c — Lorenz-96 twin experiment: EnKF vs LETKF driver.
 *
 * 1. Spin-up truth from a perturbed-F initial condition.
 * 2. Generate the initial ensemble (same for both filters).
 * 3. Build the observation network.
 * 4. Cycling: forecast -> observe -> analysis -> diagnostics.
 * 5. Write metrics CSV and state-history CSV.
 *
 * The numerical kernels live in the modules:
 *   rng.c     - random number generation
 *   l96.c     - Lorenz-96 forward model (RK4)
 *   linalg.c  - matrix inverse, Jacobi eigensolver, Gaspari-Cohn taper
 *   filter.c  - EnKF (stochastic, localized gain) and LETKF analyses
 *   diag.c    - diagnostics and CSV output helpers
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "rng.h"
#include "l96.h"
#include "linalg.h"
#include "filter.h"
#include "diag.h"

/* --- model parameters (standard Lorenz-96 benchmark) --- */
#define NX         40          /* number of state variables                 */
#define F_FORCE    8.0         /* external forcing                          */
#define DT_MODEL   0.05        /* RK4 time step (1 step ~ 6 h)              */
#define SPINUP     1000        /* number of spin-up steps (50 time units)   */

/* --- ensemble / DA parameters --- */
/* ensemble size; small ensembles are viable only thanks to localization */
#define NE         40
#define NOBS       10          /* number of observations (every 4th var)    */
#define OBS_EVERY  4
#define NCYCLES    200         /* assimilation cycles                       */
#define ASSIM_WINDOW 0.05      /* forecast window length (1 step)           */
#define OBS_STD    1.0         /* observation error standard deviation      */
#define SIGMA_INIT 0.01        /* initial ensemble perturbation std         */

/* Multiplicative prior covariance inflation (1.0 = none). */
#define INFLATION  1.0

/* Gaspari-Cohn localization half-width in grid points (cyclic distance);
 * 0.0 disables localization (global filter). */
#define LOC_RADIUS 3.0

#define OUT_DIR   "output"

int main(void) {
  /* parameters */
  const int n = NX, m = NOBS, Ne = NE;
  const double F = F_FORCE, dt = DT_MODEL;
  const double obs_std = OBS_STD;
  const double sigma_init = SIGMA_INIT;
  const int ncycles = NCYCLES;
  const double assim_window = ASSIM_WINDOW;
  const double loc_radius = LOC_RADIUS;

  /* RNG streams (distinct seeds) */
  Rng rng_truth, rng_init, rng_obs, rng_enkf_pert;
  rng_seed(&rng_truth, 2026091201ULL);
  rng_seed(&rng_init, 2026091202ULL);
  rng_seed(&rng_obs, 2026091203ULL);
  rng_seed(&rng_enkf_pert, 2026091204ULL);

  /* observation indices */
  int obs_idx[NOBS];
  build_observation_network(obs_idx, m, OBS_EVERY);

  /* output directory */
  if (!make_output_dir(OUT_DIR)) {
    fprintf(stderr, "cannot create output directory %s\n", OUT_DIR);
    return 1;
  }
  write_obs_config(obs_idx, m, OUT_DIR);

  /* allocate state vectors */
  size_t n_bytes = (size_t)n * sizeof(double);
  size_t ens_bytes = (size_t)Ne * n * sizeof(double);
  double *truth = (double *)malloc(n_bytes);
  double *ens_enkf = (double *)malloc(ens_bytes);
  double *ens_letkf = (double *)malloc(ens_bytes);
  double *mean_enkf = (double *)malloc(n_bytes);
  double *mean_letkf = (double *)malloc(n_bytes);
  double *scratch = (double *)malloc((size_t)5 * n * sizeof(double));
  double y[NOBS];
  int truth_steps, enkf_steps, letkf_steps;

  if (!truth || !ens_enkf || !ens_letkf || !mean_enkf || !mean_letkf ||
      !scratch) {
    fprintf(stderr, "top-level allocation failed\n");
    return 1;
  }

  /* ---- spin-up truth ---- */
  for (int k = 0; k < n; ++k)
    truth[k] = F + 0.01 * (2.0 * rng_uniform01(&rng_truth) - 1.0);
  l96_model_run(truth, n, F, dt, (double)SPINUP * dt, scratch);
  printf("truth spin-up: %d steps (%.1f time units)\n", SPINUP,
         (double)SPINUP * dt);

  /* ---- initial ensemble (same for both filters) ---- */
  for (int e = 0; e < Ne; ++e) {
    for (int k = 0; k < n; ++k) {
      double pert = sigma_init * rng_normal(&rng_init);
      ens_enkf[e * n + k] = truth[k] + pert;
      ens_letkf[e * n + k] = truth[k] + pert;
    }
  }

  /* ---- open output files ---- */
  char path[256];
  snprintf(path, sizeof(path), OUT_DIR "/metrics.csv");
  FILE *metrics = fopen(path, "w");
  if (!metrics) { fprintf(stderr, "cannot open %s\n", path); return 1; }

  snprintf(path, sizeof(path), OUT_DIR "/state_history.csv");
  FILE *hist = fopen(path, "w");
  if (!hist) { fprintf(stderr, "cannot open %s\n", path); return 1; }

  fprintf(metrics,
    "cycle,time,"
    "enkf_rmse_f,enkf_rmse_a,"
    "letkf_rmse_f,letkf_rmse_a,"
    "enkf_spread_f,enkf_spread_a,"
    "letkf_spread_f,letkf_spread_a,"
    "enkf_innov_rms_f,letkf_innov_rms_f\n");

  fprintf(hist, "cycle,k,truth,enkf_mean,letkf_mean\n");

  /* ---- cycle 0 diagnostics ---- */
  double t = 0.0;
  ensemble_mean(mean_enkf, ens_enkf, Ne, n);
  ensemble_mean(mean_letkf, ens_letkf, Ne, n);
  double rmse_0 = compute_rmse(mean_enkf, truth, n);
  double spread_0 = compute_spread(ens_enkf, mean_enkf, Ne, n);
  fprintf(metrics, "0,%.10f,%.10e,%.10e,%.10e,%.10e,"
                   "%.10e,%.10e,%.10e,%.10e,0,0\n",
          t, rmse_0, rmse_0, rmse_0, rmse_0,
          spread_0, spread_0, spread_0, spread_0);
  write_state_history(hist, 0, n, truth, mean_enkf, mean_letkf);

  /* ---- print header ---- */
  printf("\nLorenz-96 twin experiment: EnKF vs LETKF\n");
  printf("N=%d  F=%.1f  dt=%.2f  Ne=%d  obs=%d  window=%.2f  obs_std=%.1f"
         "  sigma_init=%.4f  loc_radius=%.1f\n",
         n, F, dt, Ne, m, assim_window, obs_std, sigma_init, loc_radius);
  printf("output directory: %s\n\n", OUT_DIR);
  printf("cycle time     EnKF_f     EnKF_a     LETKF_f    LETKF_a\n");

  /* ---- cycling ---- */
  double rmse_enkf_acc = 0.0, rmse_letkf_acc = 0.0;
  int count_ss = 0;

  for (int cycle = 1; cycle <= ncycles; ++cycle) {
    /* forecast */
    truth_steps = l96_model_run(truth, n, F, dt, assim_window, scratch);
    enkf_steps = advance_ensemble(ens_enkf, Ne, n, F, dt, assim_window,
                                  scratch);
    letkf_steps = advance_ensemble(ens_letkf, Ne, n, F, dt, assim_window,
                                   scratch);
    (void)truth_steps; (void)enkf_steps; (void)letkf_steps;

    t += assim_window;

    /* ensemble means */
    ensemble_mean(mean_enkf, ens_enkf, Ne, n);
    ensemble_mean(mean_letkf, ens_letkf, Ne, n);

    double rmse_enkf_f = compute_rmse(mean_enkf, truth, n);
    double rmse_letkf_f = compute_rmse(mean_letkf, truth, n);
    double spread_enkf_f = compute_spread(ens_enkf, mean_enkf, Ne, n);
    double spread_letkf_f = compute_spread(ens_letkf, mean_letkf, Ne, n);

    /* observations */
    make_observations(truth, obs_idx, m, obs_std, y, &rng_obs);
    double innov_enkf = obs_innovation_rms(mean_enkf, obs_idx, y, m);
    double innov_letkf = obs_innovation_rms(mean_letkf, obs_idx, y, m);

    /* covariance inflation (applied before each analysis) */
    if (INFLATION > 1.0) {
      inflate_ensemble(ens_enkf, mean_enkf, Ne, n, INFLATION);
      inflate_ensemble(ens_letkf, mean_letkf, Ne, n, INFLATION);
    }

    /* analysis */
    enkf_analysis(ens_enkf, Ne, n, obs_idx, m, y, obs_std,
                  loc_radius, &rng_enkf_pert);
    letkf_analysis(ens_letkf, Ne, n, obs_idx, m, y, obs_std, loc_radius);

    /* post-analysis diagnostics */
    ensemble_mean(mean_enkf, ens_enkf, Ne, n);
    ensemble_mean(mean_letkf, ens_letkf, Ne, n);
    double rmse_enkf_a = compute_rmse(mean_enkf, truth, n);
    double rmse_letkf_a = compute_rmse(mean_letkf, truth, n);
    double spread_enkf_a = compute_spread(ens_enkf, mean_enkf, Ne, n);
    double spread_letkf_a = compute_spread(ens_letkf, mean_letkf, Ne, n);

    /* accumulate steady-state stats (last 50 cycles) */
    if (cycle > ncycles - 50) {
      rmse_enkf_acc += rmse_enkf_a;
      rmse_letkf_acc += rmse_letkf_a;
      count_ss++;
    }

    /* write metrics row */
    fprintf(metrics,
      "%d,%.10f,%.10e,%.10e,%.10e,%.10e,"
      "%.10e,%.10e,%.10e,%.10e,%.10e,%.10e\n",
      cycle, t,
      rmse_enkf_f, rmse_enkf_a, rmse_letkf_f, rmse_letkf_a,
      spread_enkf_f, spread_enkf_a, spread_letkf_f, spread_letkf_a,
      innov_enkf, innov_letkf);

    /* write state history (analysis means) */
    write_state_history(hist, cycle, n, truth, mean_enkf, mean_letkf);

    /* console progress */
    printf("%5d %.3f  %.4e  %.4e  %.4e  %.4e\n",
           cycle, t, rmse_enkf_f, rmse_enkf_a, rmse_letkf_f, rmse_letkf_a);

    if (!isfinite(rmse_enkf_a) || !isfinite(rmse_letkf_a)) {
      fprintf(stderr, "non-finite RMSE at cycle %d\n", cycle);
      break;
    }
  }

  fclose(metrics);
  fclose(hist);

  /* summary */
  if (count_ss > 0) {
    rmse_enkf_acc /= count_ss;
    rmse_letkf_acc /= count_ss;
    printf("\nSteady-state mean analysis RMSE (last %d cycles):\n", count_ss);
    printf("  EnKF: %.4f\n", rmse_enkf_acc);
    printf("  LETKF: %.4f\n", rmse_letkf_acc);
  }

  /* --- free --- */
  free(truth); free(ens_enkf); free(ens_letkf);
  free(mean_enkf); free(mean_letkf); free(scratch);

  printf("\nDone. See %s/metrics.csv and %s/state_history.csv\n", OUT_DIR,
         OUT_DIR);
  return 0;
}