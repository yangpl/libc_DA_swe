/* filter.c — ensemble Kalman filters with Gaspari-Cohn localization.
 *
 *   EnKF  : stochastic, perturbed-observation filter with localized gain
 *           (Burgers 1998; Houtekamer & Mitchell 2001).
 *   LETKF : deterministic square-root filter in ensemble subspace,
 *           applied per state variable with local observation weights
 *           (Whitaker & Hamill 2002; Hunt et al. 2007).
 *
 * Both reduce to their global (unlocalized) form when loc_radius <= 0.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "filter.h"
#include "linalg.h"

/* Multiplicative covariance inflation: spread member anomalies about the
 * ensemble mean by 'factor'.  Compensates variance lost to sampling error
 * and to the analysis itself. */
void inflate_ensemble(double *ens, const double *mean, int Ne, int n,
                      double factor) {
  for (int e = 0; e < Ne; ++e)
    for (int k = 0; k < n; ++k)
      ens[e * n + k] = mean[k] + factor * (ens[e * n + k] - mean[k]);
}

/* ========================================================================
 * EnKF analysis (stochastic, perturbed observations; Burgers 1998)
 * ========================================================================
 * Global filter:  K = P H^T (H P H^T + R)^{-1}
 * Localized (Houtekamer & Mitchell 2001):
 *                 K = (rho . P H^T) [ H (rho . P H^T) + R ]^{-1}
 * with rho the Gaspari-Cohn taper of each state-observation pair.
 *
 * x^{a,(e)} = x^{f,(e)} + K (y + eps^{(e)} - H x^{f,(e)}),  eps ~ N(0,R)
 *
 * Returns 0 (no clipping needed for Lorenz-96).
 */
int enkf_analysis(double *ens, int Ne, int n,
                  const int *obs_idx, int m,
                  const double *y, double obs_std,
                  double loc_radius, Rng *rng_pert) {
  double inv_nm1 = 1.0 / (Ne - 1);
  double Rval = obs_std * obs_std;
  double *xmean   = (double *)calloc((size_t)n, sizeof(double));
  double *ymean   = (double *)calloc((size_t)m, sizeof(double));
  double *Y       = (double *)malloc((size_t)Ne * m * sizeof(double));
  double *Pxy_raw = (double *)calloc((size_t)n * m, sizeof(double));
  double *Cxy     = (double *)calloc((size_t)n * m, sizeof(double));
  double *S       = (double *)calloc((size_t)m * m, sizeof(double));
  double *Sinv    = (double *)calloc((size_t)m * m, sizeof(double));
  double *K       = (double *)calloc((size_t)n * m, sizeof(double));
  double *innov   = (double *)malloc((size_t)m * sizeof(double));
  double *wcol    = (double *)malloc((size_t)m * sizeof(double));

  if (!xmean || !ymean || !Y || !Pxy_raw || !Cxy || !S || !Sinv || !K ||
      !innov || !wcol) {
    fprintf(stderr, "EnKF allocation failed\n");
    exit(1);
  }

  /* --- ensemble means and raw observations --- */
  for (int e = 0; e < Ne; ++e)
    for (int k = 0; k < n; ++k)
      xmean[k] += ens[e * n + k];
  for (int k = 0; k < n; ++k) xmean[k] /= Ne;

  for (int e = 0; e < Ne; ++e)
    for (int p = 0; p < m; ++p) {
      double val = ens[e * n + obs_idx[p]];
      Y[e * m + p] = val;
      ymean[p] += val;
    }
  for (int p = 0; p < m; ++p) ymean[p] /= Ne;

  /* --- raw (unnormalised) cross-covariance  A Y_pert^T  (n x m) --- */
  for (int e = 0; e < Ne; ++e) {
    for (int p = 0; p < m; ++p) {
      double dy = Y[e * m + p] - ymean[p];
      for (int k = 0; k < n; ++k)
        Pxy_raw[k * m + p] += (ens[e * n + k] - xmean[k]) * dy;
    }
  }

  /* --- localized cross-covariance  Cxy = (1/(Ne-1)) w . Pxy_raw --- */
  for (int i = 0; i < n; ++i) {
    if (loc_radius > 0.0)
      for (int p = 0; p < m; ++p)
        wcol[p] = gc_corr(cycl_dist(i, obs_idx[p], n), loc_radius);
    else
      for (int p = 0; p < m; ++p) wcol[p] = 1.0;

    for (int p = 0; p < m; ++p)
      Cxy[i * m + p] = inv_nm1 * wcol[p] * Pxy_raw[i * m + p];
  }

  /* --- innovation covariance  S = H Cxy + R  (m x m) --- */
  for (int p = 0; p < m; ++p) {
    for (int q = 0; q < m; ++q)
      S[p * m + q] = Cxy[obs_idx[p] * m + q];
    S[p * m + p] += Rval;
  }

  if (!invert_matrix(S, Sinv, m)) {
    fprintf(stderr, "EnKF inversion failed\n");
    exit(1);
  }

  /* --- Kalman gain  K = Cxy * Sinv  (n x m) --- */
  for (int k = 0; k < n; ++k)
    for (int q = 0; q < m; ++q) {
      double sum = 0.0;
      for (int p = 0; p < m; ++p)
        sum += Cxy[k * m + p] * Sinv[p * m + q];
      K[k * m + q] = sum;
    }

  /* --- update each member with perturbed observations --- */
  for (int e = 0; e < Ne; ++e) {
    for (int p = 0; p < m; ++p) {
      double ypert = y[p] + obs_std * rng_normal(rng_pert);
      innov[p] = ypert - Y[e * m + p];
    }
    for (int k = 0; k < n; ++k) {
      double dx = 0.0;
      for (int p = 0; p < m; ++p)
        dx += K[k * m + p] * innov[p];
      ens[e * n + k] += dx;
    }
  }

  free(xmean); free(ymean); free(Y);
  free(Pxy_raw); free(Cxy); free(S); free(Sinv); free(K);
  free(innov); free(wcol);
  return 0;
}

/* ========================================================================
 * LETKF analysis (deterministic square-root, local ensemble transform;
 *                 Whitaker & Hamill 2002; Hunt et al. 2007)
 * ========================================================================
 *
 * Global ensemble-subspace square-root (loc_radius <= 0):
 *   A (n x Ne) = raw anomalies x^{(e)} - xbar;  Y' (m x Ne) = H A.
 *   Ye = Y' / sqrt(Ne-1);   T_e = I + Ye^T Ye / R.
 *   wa = T_e^{-1} Ye^T (y - ybar) / R;    xbar^a = xbar + A wa / sqrt(Ne-1).
 *   A^a = A T_e^{-1/2}   (T_e = V L V^T,  T_e^{-1/2} = V L^{-1/2} V^T).
 *   x^{a,(e)} = xbar + A [ (T_e^{-1/2})_{.,e} + wa / sqrt(Ne-1) ].
 *
 * Localized (Hunt et al. 2007): the transform is recomputed for each state
 * variable i, with every observation weighted by the Gaspari-Cohn taper
 * w_p(i) = GC(dist(i, obs_p)) (R-localization):
 *   T_e(i) = I + sum_p [w_p(i)/R] Ye_p Ye_p^T
 *   wa(i)  = T_e(i)^{-1} sum_p [w_p(i)/R] Ye_p (y_p - ybar_p)
 * and applied only to row i of the anomaly matrix A.
 *
 * Returns 0.
 */
int letkf_analysis(double *ens, int Ne, int n,
                   const int *obs_idx, int m,
                   const double *y, double obs_std,
                   double loc_radius) {
  double inv_ne = 1.0 / Ne;
  double inv_sqrt_nm1 = 1.0 / sqrt((double)(Ne - 1));
  double Rval = obs_std * obs_std;

  double *xmean  = (double *)calloc((size_t)n, sizeof(double));
  double *ymean  = (double *)calloc((size_t)m, sizeof(double));
  double *A      = (double *)malloc((size_t)n * Ne * sizeof(double));
  double *Yanom  = (double *)malloc((size_t)Ne * m * sizeof(double));
  double *T      = (double *)calloc((size_t)Ne * Ne, sizeof(double));
  double *Tinv   = (double *)calloc((size_t)Ne * Ne, sizeof(double));
  double *Teigvec = (double *)calloc((size_t)Ne * Ne, sizeof(double));
  double *Teigval = (double *)malloc((size_t)Ne * sizeof(double));
  double *Tsqrtinv = (double *)calloc((size_t)Ne * Ne, sizeof(double));
  double *rhs    = (double *)calloc((size_t)Ne, sizeof(double));
  double *wa     = (double *)calloc((size_t)Ne, sizeof(double));
  double *wcol   = (double *)malloc((size_t)m * sizeof(double));

  if (!xmean || !ymean || !A || !Yanom || !T || !Tinv || !Teigvec ||
      !Teigval || !Tsqrtinv || !rhs || !wa || !wcol) {
    fprintf(stderr, "LETKF allocation failed\n");
    exit(1);
  }

  /* --- ensemble means --- */
  for (int e = 0; e < Ne; ++e)
    for (int k = 0; k < n; ++k)
      xmean[k] += ens[e * n + k];
  for (int k = 0; k < n; ++k) xmean[k] *= inv_ne;

  for (int e = 0; e < Ne; ++e)
    for (int p = 0; p < m; ++p)
      ymean[p] += ens[e * n + obs_idx[p]];
  for (int p = 0; p < m; ++p) ymean[p] *= inv_ne;

  /* --- raw anomalies and scaled observation anomalies --- */
  for (int e = 0; e < Ne; ++e) {
    for (int k = 0; k < n; ++k)
      A[k * Ne + e] = ens[e * n + k] - xmean[k];
    for (int p = 0; p < m; ++p)
      Yanom[e * m + p] = (ens[e * n + obs_idx[p]] - ymean[p]) * inv_sqrt_nm1;
  }

  /* --- local analysis, one state variable at a time --- */
  for (int i = 0; i < n; ++i) {
    /* Gaspari-Cohn weights of every observation relative to state i */
    if (loc_radius > 0.0)
      for (int p = 0; p < m; ++p)
        wcol[p] = gc_corr(cycl_dist(i, obs_idx[p], n), loc_radius);
    else
      for (int p = 0; p < m; ++p) wcol[p] = 1.0;

    /* T_e(i) = I + sum_p [wcol[p]/R] Yanom_p Yanom_p^T  (symmetric Ne x Ne) */
    for (int a = 0; a < Ne; ++a)
      for (int b = 0; b < Ne; ++b)
        T[a * Ne + b] = (a == b) ? 1.0 : 0.0;
    for (int p = 0; p < m; ++p) {
      double wp = wcol[p];
      if (wp <= 0.0) continue;
      wp /= Rval;
      for (int a = 0; a < Ne; ++a) {
        double ya = Yanom[a * m + p];
        if (ya == 0.0) continue;
        for (int b = a; b < Ne; ++b) {
          double add = wp * ya * Yanom[b * m + p];
          T[a * Ne + b] += add;
          if (b != a) T[b * Ne + a] += add;
        }
      }
    }

    /* --- invert T_e(i) --- */
    if (!invert_matrix(T, Tinv, Ne)) {
      fprintf(stderr, "LETKF inversion failed\n");
      exit(1);
    }

    /* --- mean-shift weight for this state --- */
    for (int a = 0; a < Ne; ++a) {
      double sum = 0.0;
      for (int p = 0; p < m; ++p)
        if (wcol[p] > 0.0)
          sum += (wcol[p] / Rval) * Yanom[a * m + p] * (y[p] - ymean[p]);
      rhs[a] = sum;
    }
    for (int a = 0; a < Ne; ++a) {
      double sum = 0.0;
      for (int b = 0; b < Ne; ++b) sum += Tinv[a * Ne + b] * rhs[b];
      wa[a] = sum;
    }

    /* --- eigen-decomposition  T_e(i) = V L V^T  (Jacobi) --- */
    if (!symmetric_eigen_jacobi(T, Teigvec, Teigval, Ne)) {
      fprintf(stderr, "LETKF eigendecomposition failed\n");
      exit(1);
    }
    for (int a = 0; a < Ne; ++a)
      if (Teigval[a] < 1.0e-14) Teigval[a] = 1.0e-14;

    /* --- T_e(i)^{-1/2} = V L^{-1/2} V^T --- */
    for (int a = 0; a < Ne; ++a)
      for (int b = 0; b < Ne; ++b) {
        double sum = 0.0;
        for (int l = 0; l < Ne; ++l)
          sum += Teigvec[a * Ne + l] * (1.0 / sqrt(Teigval[l]))
                 * Teigvec[b * Ne + l];
        Tsqrtinv[a * Ne + b] = sum;
      }

    /* --- update state i of every member --- */
    for (int e = 0; e < Ne; ++e) {
      double newx = xmean[i];
      for (int j = 0; j < Ne; ++j) {
        double coeff = Tsqrtinv[j * Ne + e] + wa[j] * inv_sqrt_nm1;
        newx += A[i * Ne + j] * coeff;
      }
      ens[e * n + i] = newx;
    }
  }

  free(xmean); free(ymean); free(A); free(Yanom);
  free(T); free(Tinv); free(Teigvec); free(Teigval);
  free(Tsqrtinv); free(rhs); free(wa); free(wcol);
  return 0;
}