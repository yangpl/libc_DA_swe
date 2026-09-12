/* diag.c — diagnostics and CSV output helpers. */

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#  include <direct.h>
#else
#  include <sys/stat.h>
#endif

#include "diag.h"

void ensemble_mean(double *mean, const double *ens, int Ne, int n) {
  for (int k = 0; k < n; ++k) mean[k] = 0.0;
  for (int e = 0; e < Ne; ++e)
    for (int k = 0; k < n; ++k)
      mean[k] += ens[e * n + k];
  for (int k = 0; k < n; ++k) mean[k] /= Ne;
}

double compute_rmse(const double *x, const double *truth, int n) {
  double s = 0.0;
  for (int k = 0; k < n; ++k) {
    double d = x[k] - truth[k];
    s += d * d;
  }
  return sqrt(s / n);
}

double compute_spread(const double *ens, const double *mean, int Ne, int n) {
  double s = 0.0;
  for (int k = 0; k < n; ++k) {
    double var = 0.0;
    for (int e = 0; e < Ne; ++e) {
      double d = ens[e * n + k] - mean[k];
      var += d * d;
    }
    s += var / (Ne - 1);
  }
  return sqrt(s / n);
}

double obs_innovation_rms(const double *mean, const int *obs_idx,
                          const double *y, int m) {
  double s = 0.0;
  for (int p = 0; p < m; ++p) {
    double d = y[p] - mean[obs_idx[p]];
    s += d * d;
  }
  return sqrt(s / m);
}

void build_observation_network(int *obs_idx, int m, int every) {
  for (int p = 0; p < m; ++p)
    obs_idx[p] = p * every;
}

void make_observations(const double *truth, const int *obs_idx,
                       int m, double obs_std, double *y, Rng *rng) {
  for (int p = 0; p < m; ++p)
    y[p] = truth[obs_idx[p]] + obs_std * rng_normal(rng);
}

int make_output_dir(const char *path) {
#ifdef _WIN32
  if (_mkdir(path) == 0 || errno == EEXIST) return 1;
#else
  if (mkdir(path, 0777) == 0 || errno == EEXIST) return 1;
#endif
  return 0;
}

void write_obs_config(const int *obs_idx, int m, const char *out_dir) {
  char path[256];
  snprintf(path, sizeof(path), "%s/obs_config.csv", out_dir);
  FILE *fp = fopen(path, "w");
  if (!fp) return;

  fprintf(fp, "p,state_index\n");
  for (int p = 0; p < m; ++p)
    fprintf(fp, "%d,%d\n", p, obs_idx[p]);
  fclose(fp);
}

void write_state_history(FILE *fp, int cycle, int n,
                         const double *truth,
                         const double *sto_mean,
                         const double *det_mean) {
  for (int k = 0; k < n; ++k)
    fprintf(fp, "%d,%d,%.15e,%.15e,%.15e\n",
            cycle, k, truth[k], sto_mean[k], det_mean[k]);
}