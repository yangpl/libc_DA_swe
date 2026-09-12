/* linalg.c — dense linear algebra and covariance localization.
 *
 * The matrix inverse and symmetric Jacobi eigensolver are the same
 * routines used by the swe2d_enkf framework.  The Gaspari-Cohn taper
 * provides the covariance localization for the EnKF and LETKF. */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "linalg.h"

int invert_matrix(double *A, double *Ainv, int m) {
  double *B = (double *)malloc((size_t)m * m * sizeof(double));
  if (!B) return 0;

  for (int i = 0; i < m * m; ++i) {
    Ainv[i] = A[i];
    B[i] = 0.0;
  }
  for (int i = 0; i < m; ++i) B[i * m + i] = 1.0;

  for (int k = 0; k < m; ++k) {
    int pivot = k;
    double amax = fabs(Ainv[k * m + k]);

    for (int i = k + 1; i < m; ++i) {
      double v = fabs(Ainv[i * m + k]);
      if (v > amax) { amax = v; pivot = i; }
    }
    if (amax < 1.0e-14) { free(B); return 0; }

    if (pivot != k) {
      for (int j = 0; j < m; ++j) {
        double tmp = Ainv[k * m + j];
        Ainv[k * m + j] = Ainv[pivot * m + j];
        Ainv[pivot * m + j] = tmp;
        tmp = B[k * m + j];
        B[k * m + j] = B[pivot * m + j];
        B[pivot * m + j] = tmp;
      }
    }
    {
      double diag = Ainv[k * m + k];
      for (int j = 0; j < m; ++j) {
        Ainv[k * m + j] /= diag;
        B[k * m + j] /= diag;
      }
    }
    for (int i = 0; i < m; ++i) {
      if (i == k) continue;
      double factor = Ainv[i * m + k];
      for (int j = 0; j < m; ++j) {
        Ainv[i * m + j] -= factor * Ainv[k * m + j];
        B[i * m + j] -= factor * B[k * m + j];
      }
    }
  }

  memcpy(Ainv, B, (size_t)m * m * sizeof(double));
  free(B);
  return 1;
}

int symmetric_eigen_jacobi(const double *A, double *V, double *eval, int n) {
  double *B = (double *)malloc((size_t)n * n * sizeof(double));
  if (!B) return 0;

  memcpy(B, A, (size_t)n * n * sizeof(double));
  for (int i = 0; i < n * n; ++i) V[i] = 0.0;
  for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;

  for (int iter = 0; iter < 50 * n * n; ++iter) {
    int p = 0, q = 1;
    double max_off = 0.0;
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) {
        double aij = fabs(B[i * n + j]);
        if (aij > max_off) { max_off = aij; p = i; q = j; }
      }
    if (max_off < 1.0e-12) break;

    {
      double app = B[p * n + p], aqq = B[q * n + q], apq = B[p * n + q];
      double tau = (aqq - app) / (2.0 * apq);
      double t = ((tau >= 0.0) ? 1.0 : -1.0) /
                 (fabs(tau) + sqrt(1.0 + tau * tau));
      double c = 1.0 / sqrt(1.0 + t * t);
      double s = t * c;

      for (int k = 0; k < n; ++k) {
        if (k != p && k != q) {
          double bkp = B[k * n + p], bkq = B[k * n + q];
          double new_kp = c * bkp - s * bkq;
          double new_kq = s * bkp + c * bkq;
          B[k * n + p] = new_kp;  B[p * n + k] = new_kp;
          B[k * n + q] = new_kq;  B[q * n + k] = new_kq;
        }
      }
      B[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
      B[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
      B[p * n + q] = 0.0;  B[q * n + p] = 0.0;

      for (int k = 0; k < n; ++k) {
        double vkp = V[k * n + p], vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq;
        V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    eval[i] = B[i * n + i];
    if (!isfinite(eval[i])) { free(B); return 0; }
  }
  free(B);
  return 1;
}

/* Gaspari-Cohn (1999) fifth-order piecewise-polynomial correlation
 * function, compactly supported on [0, 2c] with c = lam / sqrt(3/10):
 *
 *   GC(z) = -1/4 z^5 + 1/2 z^4 + 5/8 z^3 - 5/3 z^2 + 1,          0 <= z <= 1
 *            1/12 z^5 - 1/2 z^4 + 5/8 z^3 + 5/3 z^2 - 5 z + 4
 *            - 2/(3 z),                                          1 <  z <= 2
 *            0,                                                   z > 2
 *   with z = |d| / c  (d = cyclic distance in grid points, lam = half-width).
 */
double gc_corr(double d, double lam) {
  double c, z, z2, z3, z4, z5;

  if (lam <= 0.0) return 1.0;          /* localization disabled */
  c = lam / sqrt(3.0 / 10.0);
  z = fabs(d) / c;
  if (z > 2.0) return 0.0;

  z2 = z * z; z3 = z2 * z; z4 = z3 * z; z5 = z4 * z;
  if (z <= 1.0)
    return -0.25 * z5 + 0.5 * z4 + 0.625 * z3 - (5.0 / 3.0) * z2 + 1.0;
  return (1.0 / 12.0) * z5 - 0.5 * z4 + 0.625 * z3
         + (5.0 / 3.0) * z2 - 5.0 * z + 4.0 - (2.0 / 3.0) / z;
}

double cycl_dist(int i, int j, int n) {
  int d = i - j;
  if (d < 0) d = -d;
  if (d > n - d) d = n - d;
  return (double)d;
}