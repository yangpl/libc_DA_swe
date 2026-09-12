#ifndef DIAG_H
#define DIAG_H

#include <stdio.h>
#include "rng.h"

/* Ensemble mean of a 1D flat ensemble [member, state]. */
void ensemble_mean(double *mean, const double *ens, int Ne, int n);

/* Domain root-mean-square difference between two states. */
double compute_rmse(const double *x, const double *truth, int n);

/* Ensemble spread (RMS of per-state sample std-dev). */
double compute_spread(const double *ens, const double *mean, int Ne, int n);

/* RMS observation innovation of the ensemble mean. */
double obs_innovation_rms(const double *mean, const int *obs_idx,
                          const double *y, int m);

/* Build observation index array (every 'every'-th state). */
void build_observation_network(int *obs_idx, int m, int every);

/* Sample noisy observations from the truth. */
void make_observations(const double *truth, const int *obs_idx,
                       int m, double obs_std, double *y, Rng *rng);

/* Create the output directory (returns 1 on success). */
int make_output_dir(const char *path);

/* Write observation-config CSV. */
void write_obs_config(const int *obs_idx, int m, const char *out_dir);

/* Append one cycle row to the state-history CSV. */
void write_state_history(FILE *fp, int cycle, int n,
                         const double *truth,
                         const double *sto_mean,
                         const double *det_mean);

#endif