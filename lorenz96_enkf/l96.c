/* l96.c — Lorenz-96 forward model and time integration.
 *
 *   dx_k/dt = x_{k-1} (x_{k+1} - x_{k-2}) - x_k + F,  cyclic indices.
 *
 * Integrated with 4th-order Runge–Kutta (Lorenz 1996). */

#include "l96.h"

/* Cyclic index modulo n (handles negative properly). */
static int wrap(int k, int n) {
  int r = k % n;
  return r < 0 ? r + n : r;
}

/* RHS of the Lorenz-96 system at index k (cyclic). */
static double l96_rhs(const double *x, int n, int k, double F) {
  int km1 = wrap(k - 1, n);
  int kp1 = wrap(k + 1, n);
  int km2 = wrap(k - 2, n);
  return x[km1] * (x[kp1] - x[km2]) - x[k] + F;
}

void l96_step_rk4(double *x, int n, double F, double dt, double *scratch) {
  double *k1 = scratch;
  double *k2 = scratch + n;
  double *k3 = scratch + 2 * n;
  double *k4 = scratch + 3 * n;
  double *tmp = scratch + 4 * n;
  int k;

  for (k = 0; k < n; ++k) k1[k] = l96_rhs(x, n, k, F);

  for (k = 0; k < n; ++k) tmp[k] = x[k] + 0.5 * dt * k1[k];
  for (k = 0; k < n; ++k) k2[k] = l96_rhs(tmp, n, k, F);

  for (k = 0; k < n; ++k) tmp[k] = x[k] + 0.5 * dt * k2[k];
  for (k = 0; k < n; ++k) k3[k] = l96_rhs(tmp, n, k, F);

  for (k = 0; k < n; ++k) tmp[k] = x[k] + dt * k3[k];
  for (k = 0; k < n; ++k) k4[k] = l96_rhs(tmp, n, k, F);

  for (k = 0; k < n; ++k)
    x[k] += (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
}

int l96_model_run(double *x, int n, double F, double dt,
                  double duration, double *scratch) {
  int nsteps = 0;
  double elapsed = 0.0;

  while (elapsed < duration - 1.0e-14) {
    double h = dt;
    if (elapsed + h > duration) h = duration - elapsed;
    l96_step_rk4(x, n, F, h, scratch);
    elapsed += h;
    nsteps++;
  }
  return nsteps;
}

int advance_ensemble(double *ens, int Ne, int n, double F, double dt,
                     double duration, double *scratch) {
  int total = 0;
  for (int e = 0; e < Ne; ++e) {
    int ns = l96_model_run(ens + e * n, n, F, dt, duration, scratch);
    if (ns < 0) return -1;
    total += ns;
  }
  return total;
}