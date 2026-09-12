#ifndef FILTER_H
#define FILTER_H

#include "rng.h"

/* EnKF (stochastic, localized gain; Houtekamer & Mitchell 2001).
 * loc_radius <= 0  =>  global filter (no localization). */
int enkf_analysis(double *ens, int Ne, int n,
                  const int *obs_idx, int m,
                  const double *y, double obs_std,
                  double loc_radius, Rng *rng_pert);

/* LETKF (deterministic square-root, per-state local transform;
 *        Whitaker & Hamill 2002; Hunt et al. 2007).
 * loc_radius <= 0  =>  global subspace filter (no localization). */
int letkf_analysis(double *ens, int Ne, int n,
                   const int *obs_idx, int m,
                   const double *y, double obs_std,
                   double loc_radius);

/* Multiplicative prior covariance inflation. */
void inflate_ensemble(double *ens, const double *mean, int Ne, int n,
                      double factor);

#endif