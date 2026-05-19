#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#define G         9.81
#define HMIN      1.0e-6
#define PI        3.14159265358979323846

#define NX        60
#define NY        60
#define LX        1.0
#define LY        1.0
#define NOBS      16
#define NCYCLES   30

#define ASSIM_WINDOW      0.04
#define OBS_STD           0.01
#define SIGMA_H0          0.02
#define SIGMA_M0          0.005
#define VAR_STEPS_WINDOW  20
#define VAR_MAX_ITER      12
#define VAR_LINESEARCH    16
#define VAR_ARMIJO_C1     1.0e-4
#define VAR_WOLFE_C2      0.9
#define VAR_ALPHA_INIT    (1.0 / 16.0)
#define VAR_ALPHA_MIN     1.0e-8
#define VAR_ALPHA_MAX     1.0
#define VAR_ALPHA_SHRINK  0.5
#define VAR_ALPHA_GROW    2.0

#define OUT_DIR "var_output"

typedef struct {
  unsigned long long state;
  int has_spare;
  double spare;
} Rng;

typedef struct {
  int nx, ny;
  double lx, ly;
  double dx, dy;
} Grid;

typedef struct {
  double *h;
  double *hu;
  double *hv;
} Field;

typedef struct {
  double h, hu, hv;
} State;

static inline int idx2(int i, int j, int nx)
{
  return i + nx * j;
}

static inline double max2(double a, double b)
{
  return (a > b) ? a : b;
}

static inline double min2(double a, double b)
{
  return (a < b) ? a : b;
}

static inline double safe_h(double h)
{
  return (h < HMIN) ? HMIN : h;
}

static inline double signum(double x)
{
  if (x > 0.0) return 1.0;
  if (x < 0.0) return -1.0;
  return 0.0;
}

static void zero_field(Field *U, int ncell)
{
  memset(U->h, 0, (size_t)ncell * sizeof(double));
  memset(U->hu, 0, (size_t)ncell * sizeof(double));
  memset(U->hv, 0, (size_t)ncell * sizeof(double));
}

static int make_output_dir(void)
{
#ifdef _WIN32
  if (_mkdir(OUT_DIR) == 0 || errno == EEXIST) return 1;
#else
  if (mkdir(OUT_DIR, 0777) == 0 || errno == EEXIST) return 1;
#endif
  return 0;
}

static int allocate_field(Field *U, int ncell)
{
  U->h = (double *)malloc((size_t)ncell * sizeof(double));
  U->hu = (double *)malloc((size_t)ncell * sizeof(double));
  U->hv = (double *)malloc((size_t)ncell * sizeof(double));

  if (!U->h || !U->hu || !U->hv) {
    free(U->h);
    free(U->hu);
    free(U->hv);
    U->h = U->hu = U->hv = NULL;
    return 0;
  }
  return 1;
}

static void free_field(Field *U)
{
  free(U->h);
  free(U->hu);
  free(U->hv);
  U->h = U->hu = U->hv = NULL;
}

static int allocate_field_array(Field *arr, int count, int ncell)
{
  for (int i = 0; i < count; ++i) {
    if (!allocate_field(&arr[i], ncell)) {
      for (int j = 0; j < i; ++j) free_field(&arr[j]);
      return 0;
    }
  }
  return 1;
}

static void free_field_array(Field *arr, int count)
{
  for (int i = 0; i < count; ++i) free_field(&arr[i]);
}

static void copy_field(Field *dst, const Field *src, int ncell)
{
  memcpy(dst->h, src->h, (size_t)ncell * sizeof(double));
  memcpy(dst->hu, src->hu, (size_t)ncell * sizeof(double));
  memcpy(dst->hv, src->hv, (size_t)ncell * sizeof(double));
}

static int apply_physical_floor(Field *U, int ncell)
{
  int clipped = 0;

  for (int k = 0; k < ncell; ++k) {
    if (U->h[k] < HMIN || !isfinite(U->h[k]) ||
	!isfinite(U->hu[k]) || !isfinite(U->hv[k])) {
      U->h[k] = HMIN;
      U->hu[k] = 0.0;
      U->hv[k] = 0.0;
      clipped++;
    }
  }
  return clipped;
}

static void rng_seed(Rng *rng, unsigned long long seed)
{
  rng->state = seed ? seed : 1ULL;
  rng->has_spare = 0;
  rng->spare = 0.0;
}

static double rng_uniform01(Rng *rng)
{
  rng->state = rng->state * 2862933555777941757ULL + 3037000493ULL;
  return ((double)(rng->state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static double rng_normal(Rng *rng)
{
  if (rng->has_spare) {
    rng->has_spare = 0;
    return rng->spare;
  }

  {
    double u1 = rng_uniform01(rng);
    double u2 = rng_uniform01(rng);
    double r = sqrt(-2.0 * log(u1));
    double th = 2.0 * PI * u2;

    rng->spare = r * sin(th);
    rng->has_spare = 1;
    return r * cos(th);
  }
}

static void initialize_gaussian_bump(Field *U, const Grid *grid)
{
  double x0 = 0.5 * grid->lx;
  double y0 = 0.5 * grid->ly;

  for (int j = 0; j < grid->ny; ++j) {
    for (int i = 0; i < grid->nx; ++i) {
      int k = idx2(i, j, grid->nx);
      double x = (i + 0.5) * grid->dx;
      double y = (j + 0.5) * grid->dy;
      double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);

      U->h[k] = 1.0 + 0.2 * exp(-r2 / 0.02);
      U->hu[k] = 0.0;
      U->hv[k] = 0.0;
    }
  }
}

static void perturb_member(Field *U, int ncell, double sigma_h,
                           double sigma_m, Rng *rng)
{
  for (int k = 0; k < ncell; ++k) {
    U->h[k] += sigma_h * rng_normal(rng);
    U->hu[k] += sigma_m * rng_normal(rng);
    U->hv[k] += sigma_m * rng_normal(rng);
  }
  apply_physical_floor(U, ncell);
}

static State get_state_bc(const Field *U, int i, int j, int nx, int ny)
{
  State q;

  if (i < 0) i = 0;
  if (i >= nx) i = nx - 1;
  if (j < 0) j = 0;
  if (j >= ny) j = ny - 1;

  {
    int k = idx2(i, j, nx);
    q.h = U->h[k];
    q.hu = U->hu[k];
    q.hv = U->hv[k];
  }
  return q;
}

static void flux_jacobian_x(State q, double J[9])
{
  double hraw = q.h;
  double h = safe_h(hraw);
  double dhs = (hraw > HMIN) ? 1.0 : 0.0;
  double m = q.hu;
  double n = q.hv;

  J[0] = 0.0;
  J[1] = 1.0;
  J[2] = 0.0;

  J[3] = (G * h - (m * m) / (h * h)) * dhs;
  J[4] = 2.0 * m / h;
  J[5] = 0.0;

  J[6] = (-(m * n) / (h * h)) * dhs;
  J[7] = n / h;
  J[8] = m / h;
}

static void flux_jacobian_y(State q, double J[9])
{
  double hraw = q.h;
  double h = safe_h(hraw);
  double dhs = (hraw > HMIN) ? 1.0 : 0.0;
  double m = q.hu;
  double n = q.hv;

  J[0] = 0.0;
  J[1] = 0.0;
  J[2] = 1.0;

  J[3] = (-(m * n) / (h * h)) * dhs;
  J[4] = n / h;
  J[5] = m / h;

  J[6] = (G * h - (n * n) / (h * h)) * dhs;
  J[7] = 0.0;
  J[8] = 2.0 * n / h;
}

static State flux_x(State q)
{
  State f;
  double h = safe_h(q.h);
  double u = q.hu / h;
  double v = q.hv / h;

  f.h = q.hu;
  f.hu = q.hu * u + 0.5 * G * h * h;
  f.hv = q.hu * v;
  return f;
}

static State flux_y(State q)
{
  State g;
  double h = safe_h(q.h);
  double u = q.hu / h;
  double v = q.hv / h;

  g.h = q.hv;
  g.hu = q.hv * u;
  g.hv = q.hv * v + 0.5 * G * h * h;
  return g;
}

static void wavespeed_grad_x(State q, double grad[3], double *speed)
{
  double hraw = q.h;
  double h = safe_h(hraw);
  double dhs = (hraw > HMIN) ? 1.0 : 0.0;
  double u = q.hu / h;
  double su = signum(u);
  double c = sqrt(G * h);

  *speed = fabs(u) + c;
  grad[0] = dhs * (-su * q.hu / (h * h) + 0.5 * G / c);
  grad[1] = su / h;
  grad[2] = 0.0;
}

static void wavespeed_grad_y(State q, double grad[3], double *speed)
{
  double hraw = q.h;
  double h = safe_h(hraw);
  double dhs = (hraw > HMIN) ? 1.0 : 0.0;
  double v = q.hv / h;
  double sv = signum(v);
  double c = sqrt(G * h);

  *speed = fabs(v) + c;
  grad[0] = dhs * (-sv * q.hv / (h * h) + 0.5 * G / c);
  grad[1] = 0.0;
  grad[2] = sv / h;
}

static void rusanov_x(State qL, State qR, State *fout)
{
  State fL = flux_x(qL);
  State fR = flux_x(qR);
  double gradL[3], gradR[3];
  double aL, aR, s;

  wavespeed_grad_x(qL, gradL, &aL);
  wavespeed_grad_x(qR, gradR, &aR);
  s = max2(aL, aR);

  fout->h = 0.5 * (fL.h + fR.h) - 0.5 * s * (qR.h - qL.h);
  fout->hu = 0.5 * (fL.hu + fR.hu) - 0.5 * s * (qR.hu - qL.hu);
  fout->hv = 0.5 * (fL.hv + fR.hv) - 0.5 * s * (qR.hv - qL.hv);
}

static void rusanov_y(State qB, State qT, State *gout)
{
  State gB = flux_y(qB);
  State gT = flux_y(qT);
  double gradB[3], gradT[3];
  double aB, aT, s;

  wavespeed_grad_y(qB, gradB, &aB);
  wavespeed_grad_y(qT, gradT, &aT);
  s = max2(aB, aT);

  gout->h = 0.5 * (gB.h + gT.h) - 0.5 * s * (qT.h - qB.h);
  gout->hu = 0.5 * (gB.hu + gT.hu) - 0.5 * s * (qT.hu - qB.hu);
  gout->hv = 0.5 * (gB.hv + gT.hv) - 0.5 * s * (qT.hv - qB.hv);
}

static void rusanov_x_jacobians(State qL, State qR, double JL[9], double JR[9])
{
  double A_L[9], A_R[9];
  double gradL[3] = {0}, gradR[3] = {0};
  double aL, aR, s;
  double dq[3];
  const double *gsL;
  const double *gsR;

  flux_jacobian_x(qL, A_L);
  flux_jacobian_x(qR, A_R);
  wavespeed_grad_x(qL, gradL, &aL);
  wavespeed_grad_x(qR, gradR, &aR);

  dq[0] = qR.h - qL.h;
  dq[1] = qR.hu - qL.hu;
  dq[2] = qR.hv - qL.hv;

  if (aL >= aR) {
    s = aL;
    gsL = gradL;
    gsR = NULL;
  } else {
    s = aR;
    gsL = NULL;
    gsR = gradR;
  }

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double left = 0.5 * A_L[3 * r + c] + 0.5 * s * (r == c ? 1.0 : 0.0);
      double right = 0.5 * A_R[3 * r + c] - 0.5 * s * (r == c ? 1.0 : 0.0);
      if (gsL) left -= 0.5 * dq[r] * gsL[c];
      if (gsR) right -= 0.5 * dq[r] * gsR[c];
      JL[3 * r + c] = left;
      JR[3 * r + c] = right;
    }
  }
}

static void rusanov_y_jacobians(State qB, State qT, double JB[9], double JT[9])
{
  double A_B[9], A_T[9];
  double gradB[3] = {0}, gradT[3] = {0};
  double aB, aT, s;
  double dq[3];
  const double *gsB;
  const double *gsT;

  flux_jacobian_y(qB, A_B);
  flux_jacobian_y(qT, A_T);
  wavespeed_grad_y(qB, gradB, &aB);
  wavespeed_grad_y(qT, gradT, &aT);

  dq[0] = qT.h - qB.h;
  dq[1] = qT.hu - qB.hu;
  dq[2] = qT.hv - qB.hv;

  if (aB >= aT) {
    s = aB;
    gsB = gradB;
    gsT = NULL;
  } else {
    s = aT;
    gsB = NULL;
    gsT = gradT;
  }

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double bottom = 0.5 * A_B[3 * r + c] + 0.5 * s * (r == c ? 1.0 : 0.0);
      double top = 0.5 * A_T[3 * r + c] - 0.5 * s * (r == c ? 1.0 : 0.0);
      if (gsB) bottom -= 0.5 * dq[r] * gsB[c];
      if (gsT) top -= 0.5 * dq[r] * gsT[c];
      JB[3 * r + c] = bottom;
      JT[3 * r + c] = top;
    }
  }
}

static void euler_update(const Field *Uin, Field *Uout, const Grid *grid, double dt)
{
  for (int j = 0; j < grid->ny; ++j) {
    for (int i = 0; i < grid->nx; ++i) {
      int k = idx2(i, j, grid->nx);
      State qij = get_state_bc(Uin, i, j, grid->nx, grid->ny);
      State qim1j = get_state_bc(Uin, i - 1, j, grid->nx, grid->ny);
      State qip1j = get_state_bc(Uin, i + 1, j, grid->nx, grid->ny);
      State qijm1 = get_state_bc(Uin, i, j - 1, grid->nx, grid->ny);
      State qijp1 = get_state_bc(Uin, i, j + 1, grid->nx, grid->ny);
      State FxL, FxR, FyB, FyT;

      rusanov_x(qim1j, qij, &FxL);
      rusanov_x(qij, qip1j, &FxR);
      rusanov_y(qijm1, qij, &FyB);
      rusanov_y(qij, qijp1, &FyT);

      Uout->h[k] = Uin->h[k]
	- dt / grid->dx * (FxR.h - FxL.h)
	- dt / grid->dy * (FyT.h - FyB.h);
      Uout->hu[k] = Uin->hu[k]
	- dt / grid->dx * (FxR.hu - FxL.hu)
	- dt / grid->dy * (FyT.hu - FyB.hu);
      Uout->hv[k] = Uin->hv[k]
	- dt / grid->dx * (FxR.hv - FxL.hv)
	- dt / grid->dy * (FyT.hv - FyB.hv);
    }
  }
}

static void step_rk2_store(const Field *Uin, Field *Ustage1, Field *Utmp2,
                           Field *Uout, const Grid *grid, double dt)
{
  int ncell = grid->nx * grid->ny;

  euler_update(Uin, Ustage1, grid, dt);
  euler_update(Ustage1, Utmp2, grid, dt);

  for (int k = 0; k < ncell; ++k) {
    Uout->h[k] = 0.5 * Uin->h[k] + 0.5 * Utmp2->h[k];
    Uout->hu[k] = 0.5 * Uin->hu[k] + 0.5 * Utmp2->hu[k];
    Uout->hv[k] = 0.5 * Uin->hv[k] + 0.5 * Utmp2->hv[k];
  }
  apply_physical_floor(Uout, ncell);
}

static void build_observation_network(const Grid *grid, int *obs_idx, int m)
{
  int p = 0;
  int ix_start = grid->nx / 6;
  int ix_end = 5 * grid->nx / 6;
  int iy_start = grid->ny / 6;
  int iy_end = 5 * grid->ny / 6;
  int nxo = (int)(sqrt((double)m) + 0.5);
  int nyo = nxo;

  if (nxo * nyo < m) nyo += 1;

  for (int b = 0; b < nyo && p < m; ++b) {
    int j = (nyo == 1)
      ? (iy_start + iy_end) / 2
      : iy_start + (iy_end - iy_start) * b / (nyo - 1);

    for (int a = 0; a < nxo && p < m; ++a) {
      int i = (nxo == 1)
	? (ix_start + ix_end) / 2
	: ix_start + (ix_end - ix_start) * a / (nxo - 1);
      obs_idx[p++] = idx2(i, j, grid->nx);
    }
  }
}

static void make_observations(const Field *truth, const int *obs_idx,
                              int m, double obs_std, double *y, Rng *rng)
{
  for (int p = 0; p < m; ++p) {
    y[p] = truth->h[obs_idx[p]] + obs_std * rng_normal(rng);
  }
}

static double rmse_h(const Field *U, const Field *truth, int ncell)
{
  double s = 0.0;

  for (int k = 0; k < ncell; ++k) {
    double d = U->h[k] - truth->h[k];
    s += d * d;
  }
  return sqrt(s / ncell);
}

static double field_dot(const Field *A, const Field *B, int ncell)
{
  double s = 0.0;

  for (int k = 0; k < ncell; ++k) {
    s += A->h[k] * B->h[k];
    s += A->hu[k] * B->hu[k];
    s += A->hv[k] * B->hv[k];
  }
  return s;
}

static double field_norm(const Field *A, int ncell)
{
  return sqrt(field_dot(A, A, ncell));
}

static void field_axpy(Field *Y, double a, const Field *X, int ncell)
{
  for (int k = 0; k < ncell; ++k) {
    Y->h[k] += a * X->h[k];
    Y->hu[k] += a * X->hu[k];
    Y->hv[k] += a * X->hv[k];
  }
}

static void field_combine(Field *Y, const Field *A, double alpha,
                          const Field *P, int ncell)
{
  for (int k = 0; k < ncell; ++k) {
    Y->h[k] = A->h[k] + alpha * P->h[k];
    Y->hu[k] = A->hu[k] + alpha * P->hu[k];
    Y->hv[k] = A->hv[k] + alpha * P->hv[k];
  }
}

static void field_copy_scaled_precond(Field *dst, const Field *grad,
                                      double var_h, double var_m, int ncell)
{
  for (int k = 0; k < ncell; ++k) {
    dst->h[k] = -var_h * grad->h[k];
    dst->hu[k] = -var_m * grad->hu[k];
    dst->hv[k] = -var_m * grad->hv[k];
  }
}

static void obs_gradient_add(Field *lambda, const Field *state,
                             const int *obs_idx, const double *y,
                             int m, double obs_var)
{
  for (int p = 0; p < m; ++p) {
    int k = obs_idx[p];
    lambda->h[k] += (state->h[k] - y[p]) / obs_var;
  }
}

static void add_matT_vec_to_cell(Field *lam, int cell, const double J[9],
                                 const double v[3], double coeff)
{
  lam->h[cell]  += coeff * (J[0] * v[0] + J[3] * v[1] + J[6] * v[2]);
  lam->hu[cell] += coeff * (J[1] * v[0] + J[4] * v[1] + J[7] * v[2]);
  lam->hv[cell] += coeff * (J[2] * v[0] + J[5] * v[1] + J[8] * v[2]);
}

static void euler_adjoint(const Field *Uin, const Field *lam_out,
                          Field *lam_in, const Grid *grid, double dt)
{
  double JL[9], JR[9], JB[9], JT[9];

  zero_field(lam_in, grid->nx * grid->ny);

  for (int j = 0; j < grid->ny; ++j) {
    for (int i = 0; i < grid->nx; ++i) {
      int iL = (i > 0) ? (i - 1) : 0;
      int iR = (i + 1 < grid->nx) ? (i + 1) : (grid->nx - 1);
      int jB = (j > 0) ? (j - 1) : 0;
      int jT = (j + 1 < grid->ny) ? (j + 1) : (grid->ny - 1);
      int k = idx2(i, j, grid->nx);
      int kL = idx2(iL, j, grid->nx);
      int kR = idx2(iR, j, grid->nx);
      int kB = idx2(i, jB, grid->nx);
      int kT = idx2(i, jT, grid->nx);
      State qij = get_state_bc(Uin, i, j, grid->nx, grid->ny);
      State qim1j = get_state_bc(Uin, i - 1, j, grid->nx, grid->ny);
      State qip1j = get_state_bc(Uin, i + 1, j, grid->nx, grid->ny);
      State qijm1 = get_state_bc(Uin, i, j - 1, grid->nx, grid->ny);
      State qijp1 = get_state_bc(Uin, i, j + 1, grid->nx, grid->ny);
      double v[3];

      v[0] = lam_out->h[k];
      v[1] = lam_out->hu[k];
      v[2] = lam_out->hv[k];

      lam_in->h[k] += v[0];
      lam_in->hu[k] += v[1];
      lam_in->hv[k] += v[2];

      rusanov_x_jacobians(qij, qip1j, JL, JR);
      add_matT_vec_to_cell(lam_in, k,  JL, v, -dt / grid->dx);
      add_matT_vec_to_cell(lam_in, kR, JR, v, -dt / grid->dx);

      rusanov_x_jacobians(qim1j, qij, JL, JR);
      add_matT_vec_to_cell(lam_in, kL, JL, v,  dt / grid->dx);
      add_matT_vec_to_cell(lam_in, k,  JR, v,  dt / grid->dx);

      rusanov_y_jacobians(qij, qijp1, JB, JT);
      add_matT_vec_to_cell(lam_in, k,  JB, v, -dt / grid->dy);
      add_matT_vec_to_cell(lam_in, kT, JT, v, -dt / grid->dy);

      rusanov_y_jacobians(qijm1, qij, JB, JT);
      add_matT_vec_to_cell(lam_in, kB, JB, v,  dt / grid->dy);
      add_matT_vec_to_cell(lam_in, k,  JT, v,  dt / grid->dy);
    }
  }
}

static void rk2_adjoint_step(const Field *Qn, const Field *Q1,
                             const Field *lam_np1, Field *lam_n,
                             Field *lam_stage2, Field *lam_stage1,
                             const Grid *grid, double dt)
{
  int ncell = grid->nx * grid->ny;

  for (int k = 0; k < ncell; ++k) {
    lam_stage2->h[k] = 0.5 * lam_np1->h[k];
    lam_stage2->hu[k] = 0.5 * lam_np1->hu[k];
    lam_stage2->hv[k] = 0.5 * lam_np1->hv[k];

    lam_n->h[k] = 0.5 * lam_np1->h[k];
    lam_n->hu[k] = 0.5 * lam_np1->hu[k];
    lam_n->hv[k] = 0.5 * lam_np1->hv[k];
  }

  euler_adjoint(Q1, lam_stage2, lam_stage1, grid, dt);
  euler_adjoint(Qn, lam_stage1, lam_stage2, grid, dt);
  field_axpy(lam_n, 1.0, lam_stage2, ncell);
}

static void write_field_csv(const char *filename, const Field *U,
                            const Grid *grid, double time)
{
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "warning: cannot open %s\n", filename);
    return;
  }

  fprintf(fp, "# time=%.10f\n", time);
  fprintf(fp, "i,j,x,y,h,hu,hv\n");

  for (int j = 0; j < grid->ny; ++j) {
    for (int i = 0; i < grid->nx; ++i) {
      int k = idx2(i, j, grid->nx);
      double x = (i + 0.5) * grid->dx;
      double y = (j + 0.5) * grid->dy;
      fprintf(fp, "%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f\n",
	      i, j, x, y, U->h[k], U->hu[k], U->hv[k]);
    }
  }

  fclose(fp);
}

static void write_obs_config(const int *obs_idx, int m)
{
  char path[256];
  FILE *fp;

  snprintf(path, sizeof(path), OUT_DIR "/obs_config.csv");
  fp = fopen(path, "w");
  if (!fp) return;

  fprintf(fp, "p,cell_index,i,j\n");
  for (int p = 0; p < m; ++p) {
    int i = obs_idx[p] % NX;
    int j = obs_idx[p] / NX;
    fprintf(fp, "%d,%d,%d,%d\n", p, obs_idx[p], i, j);
  }
  fclose(fp);
}

static void write_cycle_triplet(int cycle, const Grid *grid, double t,
                                const Field *truth, const Field *background,
                                const Field *analysis)
{
  char path[256];

  snprintf(path, sizeof(path), OUT_DIR "/truth_cycle%03d.csv", cycle);
  write_field_csv(path, truth, grid, t);

  snprintf(path, sizeof(path), OUT_DIR "/background_cycle%03d.csv", cycle);
  write_field_csv(path, background, grid, t);

  snprintf(path, sizeof(path), OUT_DIR "/analysis_cycle%03d.csv", cycle);
  write_field_csv(path, analysis, grid, t);
}

static double evaluate_cost_gradient(
				     const Field *x0,
				     const Field *xb,
				     const Grid *grid,
				     const int *obs_idx,
				     const double *obs,
				     Field *traj,
				     Field *stage1,
				     Field *qtmp,
				     Field *lam_curr,
				     Field *lam_prev,
				     Field *lam_work,
				     Field *grad,
				     double *cycle_obs_cost,
				     double *cycle_obs_rms,
				     int nsteps_total,
				     double dt,
				     double sigma_h,
				     double sigma_m,
				     double obs_std)
{
  const double var_h = sigma_h * sigma_h;
  const double var_m = sigma_m * sigma_m;
  const double obs_var = obs_std * obs_std;
  const int ncell = grid->nx * grid->ny;
  double J = 0.0;

  if (cycle_obs_cost) {
    for (int cycle = 0; cycle < NCYCLES; ++cycle) cycle_obs_cost[cycle] = 0.0;
  }
  if (cycle_obs_rms) {
    for (int cycle = 0; cycle < NCYCLES; ++cycle) cycle_obs_rms[cycle] = 0.0;
  }

  copy_field(&traj[0], x0, ncell);
  for (int n = 0; n < nsteps_total; ++n) {
    step_rk2_store(&traj[n], &stage1[n], qtmp, &traj[n + 1], grid, dt);
  }

  zero_field(grad, ncell);
  for (int k = 0; k < ncell; ++k) {
    double dh = x0->h[k] - xb->h[k];
    double dhu = x0->hu[k] - xb->hu[k];
    double dhv = x0->hv[k] - xb->hv[k];

    J += 0.5 * dh * dh / var_h;
    J += 0.5 * dhu * dhu / var_m;
    J += 0.5 * dhv * dhv / var_m;

    grad->h[k] = dh / var_h;
    grad->hu[k] = dhu / var_m;
    grad->hv[k] = dhv / var_m;
  }

  for (int cycle = 1; cycle <= NCYCLES; ++cycle) {
    int step = cycle * VAR_STEPS_WINDOW;
    const double *y = obs + (size_t)(cycle - 1) * NOBS;
    const Field *U = &traj[step];
    double jobs = 0.0;
    double srms = 0.0;

    for (int p = 0; p < NOBS; ++p) {
      double d = U->h[obs_idx[p]] - y[p];
      J += 0.5 * d * d / obs_var;
      jobs += 0.5 * d * d / obs_var;
      srms += d * d;
    }
    if (cycle_obs_cost) cycle_obs_cost[cycle - 1] = jobs;
    if (cycle_obs_rms) cycle_obs_rms[cycle - 1] = sqrt(srms / NOBS);
  }

  zero_field(lam_curr, ncell);
  for (int n = nsteps_total; n >= 1; --n) {
    if (n % VAR_STEPS_WINDOW == 0) {
      int cycle = n / VAR_STEPS_WINDOW;
      const double *y = obs + (size_t)(cycle - 1) * NOBS;
      obs_gradient_add(lam_curr, &traj[n], obs_idx, y, NOBS, obs_var);
    }

    rk2_adjoint_step(&traj[n - 1], &stage1[n - 1],
		     lam_curr, lam_prev, lam_work, qtmp,
		     grid, dt);
    copy_field(lam_curr, lam_prev, ncell);
  }

  field_axpy(grad, 1.0, lam_curr, ncell);
  return J;
}

static int line_search_two_way(
			       const Field *x0,
			       const Field *xb,
			       const Field *direction,
			       double J,
			       double gtp,
			       double alpha_hint,
			       const Grid *grid,
			       const int *obs_idx,
			       const double *obs,
			       Field *trial0,
			       Field *traj,
			       Field *stage1,
			       Field *qtmp,
			       Field *lam_curr,
			       Field *lam_prev,
			       Field *lam_work,
			       Field *trial_grad,
			       int nsteps_total,
			       double dt,
			       double sigma_h,
			       double sigma_m,
			       double obs_std,
			       int ncell,
			       double *alpha_out)
{
  double alpha = min2(max2(alpha_hint, VAR_ALPHA_MIN), VAR_ALPHA_MAX);
  double alpha_lo = 0.0;
  double alpha_hi = 0.0;
  double trial_cost = 0.0;
  double trial_gtp = 0.0;
  int ls = 0;

  if (!(gtp < 0.0) || !isfinite(gtp)) return 0;

  while (ls < VAR_LINESEARCH) {
    field_combine(trial0, x0, alpha, direction, ncell);
    apply_physical_floor(trial0, ncell);
    trial_cost = evaluate_cost_gradient(trial0, xb, grid, obs_idx, obs,
					traj, stage1, qtmp, lam_curr, lam_prev,
					lam_work, trial_grad, NULL, NULL,
					nsteps_total, dt,
					sigma_h, sigma_m, obs_std);
    trial_gtp = field_dot(trial_grad, direction, ncell);
    ls++;

    if (isfinite(trial_cost) && isfinite(trial_gtp) &&
	trial_cost <= J + VAR_ARMIJO_C1 * alpha * gtp &&
	trial_gtp >= VAR_WOLFE_C2 * gtp) {
      *alpha_out = alpha;
      return 1;
    }

    if (!isfinite(trial_cost) || trial_cost > J + VAR_ARMIJO_C1 * alpha * gtp) {
      alpha_hi = alpha;

      if (alpha_lo > 0.0) {
	alpha = 0.5 * (alpha_lo + alpha_hi);
      } else {
	alpha *= VAR_ALPHA_SHRINK;
      }
    } else {
      alpha_lo = alpha;

      if (alpha_hi > 0.0) {
	alpha = 0.5 * (alpha_lo + alpha_hi);
      } else {
	alpha = min2(VAR_ALPHA_GROW * alpha, VAR_ALPHA_MAX);
      }
    }

    if (alpha < VAR_ALPHA_MIN || alpha > VAR_ALPHA_MAX) break;
    if ((alpha_hi > 0.0 && alpha_hi - alpha_lo < VAR_ALPHA_MIN) ||
	alpha <= alpha_lo || (alpha_hi > 0.0 && alpha >= alpha_hi)) {
      break;
    }
  }

  return 0;
}

int main(void)
{
  const int ncell = NX * NY;
  const int nsteps_total = NCYCLES * VAR_STEPS_WINDOW;
  const double dt = ASSIM_WINDOW / VAR_STEPS_WINDOW;
  const double var_h = SIGMA_H0 * SIGMA_H0;
  const double var_m = SIGMA_M0 * SIGMA_M0;
  Grid grid;
  int obs_idx[NOBS];
  double *obs = NULL;
  Rng rng_init, rng_obs;
  Field truth0, background0, analysis0, trial0;
  Field grad, trial_grad, direction, truth_tmp;
  Field lam_curr, lam_prev, lam_work;
  Field *truth_traj = NULL;
  Field *bg_traj = NULL;
  Field *an_traj = NULL;
  Field *stage = NULL;
  Field qtmp;
  FILE *hist = NULL;
  FILE *misfit_hist = NULL;
  double cycle_obs_cost[NCYCLES];
  double cycle_obs_rms[NCYCLES];
  double alpha_prev = VAR_ALPHA_INIT;

  rng_seed(&rng_init, 2026050901ULL);
  rng_seed(&rng_obs, 2026050902ULL);

  grid.nx = NX;
  grid.ny = NY;
  grid.lx = LX;
  grid.ly = LY;
  grid.dx = grid.lx / grid.nx;
  grid.dy = grid.ly / grid.ny;

  if (!make_output_dir()) {
    fprintf(stderr, "cannot create output directory %s\n", OUT_DIR);
    return 1;
  }

  obs = (double *)malloc((size_t)NCYCLES * NOBS * sizeof(double));
  truth_traj = (Field *)malloc((size_t)(nsteps_total + 1) * sizeof(Field));
  bg_traj = (Field *)malloc((size_t)(nsteps_total + 1) * sizeof(Field));
  an_traj = (Field *)malloc((size_t)(nsteps_total + 1) * sizeof(Field));
  stage = (Field *)malloc((size_t)nsteps_total * sizeof(Field));
  if (!obs || !truth_traj || !bg_traj || !an_traj || !stage) {
    fprintf(stderr, "top-level allocation failed\n");
    free(obs);
    free(truth_traj);
    free(bg_traj);
    free(an_traj);
    free(stage);
    return 1;
  }

  if (!allocate_field(&truth0, ncell) ||
      !allocate_field(&background0, ncell) ||
      !allocate_field(&analysis0, ncell) ||
      !allocate_field(&trial0, ncell) ||
      !allocate_field(&grad, ncell) ||
      !allocate_field(&trial_grad, ncell) ||
      !allocate_field(&direction, ncell) ||
      !allocate_field(&truth_tmp, ncell) ||
      !allocate_field(&lam_curr, ncell) ||
      !allocate_field(&lam_prev, ncell) ||
      !allocate_field(&lam_work, ncell) ||
      !allocate_field(&qtmp, ncell) ||
      !allocate_field_array(truth_traj, nsteps_total + 1, ncell) ||
      !allocate_field_array(bg_traj, nsteps_total + 1, ncell) ||
      !allocate_field_array(an_traj, nsteps_total + 1, ncell) ||
      !allocate_field_array(stage, nsteps_total, ncell)) {
    fprintf(stderr, "field allocation failed\n");
    return 1;
  }

  build_observation_network(&grid, obs_idx, NOBS);
  write_obs_config(obs_idx, NOBS);

  initialize_gaussian_bump(&truth0, &grid);
  copy_field(&background0, &truth0, ncell);
  perturb_member(&background0, ncell, SIGMA_H0, SIGMA_M0, &rng_init);
  copy_field(&analysis0, &background0, ncell);

  copy_field(&truth_traj[0], &truth0, ncell);
  for (int n = 0; n < nsteps_total; ++n) {
    step_rk2_store(&truth_traj[n], &stage[n], &truth_tmp, &truth_traj[n + 1], &grid, dt);
  }
  for (int cycle = 1; cycle <= NCYCLES; ++cycle) {
    make_observations(&truth_traj[cycle * VAR_STEPS_WINDOW],
		      obs_idx, NOBS, OBS_STD,
		      obs + (size_t)(cycle - 1) * NOBS, &rng_obs);
  }

  hist = fopen(OUT_DIR "/cost_history.csv", "w");
  if (!hist) {
    fprintf(stderr, "cannot open cost history\n");
    return 1;
  }
  misfit_hist = fopen(OUT_DIR "/iter_cycle_misfit.csv", "w");
  if (!misfit_hist) {
    fprintf(stderr, "cannot open iteration-cycle misfit history\n");
    return 1;
  }
  fprintf(hist, "iter,cost,grad_norm,step_length\n");
  fprintf(misfit_hist, "iter,cycle,time,obs_cost,obs_rms\n");

  printf("Strong-constraint 4D-Var with discrete adjoint\n");
  printf("grid=%dx%d obs=%d cycles=%d fixed_dt=%.6f steps=%d\n",
	 NX, NY, NOBS, NCYCLES, dt, nsteps_total);
  printf("output directory: %s\n\n", OUT_DIR);
  printf("iter cost             grad_norm        step\n");

  for (int iter = 0; iter < VAR_MAX_ITER; ++iter) {
    double J = evaluate_cost_gradient(&analysis0, &background0, &grid, obs_idx, obs,
				      an_traj, stage, &qtmp, &lam_curr, &lam_prev,
				      &lam_work, &grad, cycle_obs_cost, cycle_obs_rms,
				      nsteps_total, dt,
				      SIGMA_H0, SIGMA_M0, OBS_STD);
    double grad_norm = field_norm(&grad, ncell);
    double alpha = 0.0;

    printf("%4d %.10e %.10e ", iter, J, grad_norm);

    if (!isfinite(J) || !isfinite(grad_norm)) {
      fprintf(stderr, "non-finite cost or gradient\n");
      fclose(hist);
      return 1;
    }

    if (grad_norm < 1.0e-6) {
      fprintf(hist, "%d,%.12e,%.12e,0\n", iter, J, grad_norm);
      for (int cycle = 1; cycle <= NCYCLES; ++cycle) {
	fprintf(misfit_hist, "%d,%d,%.10f,%.12e,%.12e\n",
		iter, cycle, cycle * ASSIM_WINDOW,
		cycle_obs_cost[cycle - 1], cycle_obs_rms[cycle - 1]);
      }
      printf("0\n");
      break;
    }

    for (int cycle = 1; cycle <= NCYCLES; ++cycle) {
      fprintf(misfit_hist, "%d,%d,%.10f,%.12e,%.12e\n",
	      iter, cycle, cycle * ASSIM_WINDOW,
	      cycle_obs_cost[cycle - 1], cycle_obs_rms[cycle - 1]);
    }

    field_copy_scaled_precond(&direction, &grad, var_h, var_m, ncell);

    {
      double gtp = field_dot(&grad, &direction, ncell);
      int accepted = 0;

      accepted = line_search_two_way(&analysis0, &background0, &direction,
				     J, gtp, alpha_prev,
				     &grid, obs_idx, obs,
				     &trial0, bg_traj, stage, &qtmp,
				     &lam_curr, &lam_prev, &lam_work, &trial_grad,
				     nsteps_total, dt,
				     SIGMA_H0, SIGMA_M0, OBS_STD,
				     ncell, &alpha);

      if (!accepted) {
	fprintf(hist, "%d,%.12e,%.12e,0\n", iter, J, grad_norm);
	printf("failed\n");
	fprintf(stderr, "line search failed at iteration %d\n", iter);
	break;
      }

      copy_field(&analysis0, &trial0, ncell);
      alpha_prev = alpha;
    }

    fprintf(hist, "%d,%.12e,%.12e,%.12e\n", iter, J, grad_norm, alpha);
    printf("%.3e\n", alpha);
  }

  fclose(hist);
  fclose(misfit_hist);

  evaluate_cost_gradient(&background0, &background0, &grid, obs_idx, obs,
			 bg_traj, stage, &qtmp, &lam_curr, &lam_prev, &lam_work,
			 &grad, NULL, NULL, nsteps_total, dt, SIGMA_H0, SIGMA_M0, OBS_STD);
  evaluate_cost_gradient(&analysis0, &background0, &grid, obs_idx, obs,
			 an_traj, stage, &qtmp, &lam_curr, &lam_prev, &lam_work,
			 &grad, NULL, NULL, nsteps_total, dt, SIGMA_H0, SIGMA_M0, OBS_STD);

  {
    FILE *metrics = fopen(OUT_DIR "/trajectory_metrics.csv", "w");
    if (!metrics) {
      fprintf(stderr, "cannot open trajectory metrics\n");
      return 1;
    }

    fprintf(metrics, "cycle,time,background_rmse_h,analysis_rmse_h\n");
    write_cycle_triplet(0, &grid, 0.0, &truth_traj[0], &bg_traj[0], &an_traj[0]);

    printf("\ncycle time     background    analysis\n");
    for (int cycle = 0; cycle <= NCYCLES; ++cycle) {
      int step = cycle * VAR_STEPS_WINDOW;
      double time = cycle * ASSIM_WINDOW;
      double bg_rmse = rmse_h(&bg_traj[step], &truth_traj[step], ncell);
      double an_rmse = rmse_h(&an_traj[step], &truth_traj[step], ncell);

      fprintf(metrics, "%d,%.10f,%.10e,%.10e\n", cycle, time, bg_rmse, an_rmse);
      if (cycle > 0) {
	write_cycle_triplet(cycle, &grid, time,
			    &truth_traj[step], &bg_traj[step], &an_traj[step]);
      }
      printf("%5d %.3f  %.4e  %.4e\n", cycle, time, bg_rmse, an_rmse);
    }
    fclose(metrics);
  }

  write_field_csv(OUT_DIR "/background_initial.csv", &background0, &grid, 0.0);
  write_field_csv(OUT_DIR "/analysis_initial.csv", &analysis0, &grid, 0.0);
  write_field_csv(OUT_DIR "/truth_initial.csv", &truth0, &grid, 0.0);

  free(obs);
  free_field_array(truth_traj, nsteps_total + 1);
  free_field_array(bg_traj, nsteps_total + 1);
  free_field_array(an_traj, nsteps_total + 1);
  free_field_array(stage, nsteps_total);
  free(truth_traj);
  free(bg_traj);
  free(an_traj);
  free(stage);

  free_field(&truth0);
  free_field(&background0);
  free_field(&analysis0);
  free_field(&trial0);
  free_field(&grad);
  free_field(&trial_grad);
  free_field(&direction);
  free_field(&truth_tmp);
  free_field(&lam_curr);
  free_field(&lam_prev);
  free_field(&lam_work);
  free_field(&qtmp);

  printf("\nDone. Outputs written to %s\n", OUT_DIR);
  return 0;
}
