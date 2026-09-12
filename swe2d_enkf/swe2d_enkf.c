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
#define CFL       0.40
#define HMIN      1.0e-6
#define PI        3.14159265358979323846

#define NX        60
#define NY        60
#define LX        1.0
#define LY        1.0
#define NE        20
#define NOBS      16
#define NCYCLES   30

#define ASSIM_WINDOW 0.04
#define OBS_STD      0.01
#define SIGMA_H0     0.02
#define SIGMA_M0     0.005

#define OUT_DIR "comparison_output"

// SWE twin experiment: stochastic EnKF vs deterministic EnSRF (theory.tex)
// State U = [h, hu, hv]^T — conserved form  ∂_t U + ∂_x F(U) + ∂_y G(U) = 0
// Forward: finite-volume, Rusanov flux, 2-stage RK2, CFL time step
// EnKF:  perturbed-observation analysis (Burgers 1998)
// EnSRF: symmetric square-root in ensemble subspace (Whitaker & Hamill 2002)
// Observations: h-only at NOBS sparse interior cells, σ_obs = OBS_STD

typedef struct { unsigned long long state; int has_spare; double spare; } Rng;

typedef struct { int nx, ny; double lx, ly, dx, dy; } Grid;

typedef struct { double *h, *hu, *hv; } Field;

typedef struct { double h, hu, hv; } State;

static void rng_seed(Rng *rng, unsigned long long seed) {
  rng->state = seed ? seed : 1ULL;
  rng->has_spare = 0;
  rng->spare = 0.0;
}

static double rng_uniform01(Rng *rng) {
  rng->state = rng->state * 2862933555777941757ULL + 3037000493ULL;
  return ((double)(rng->state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static double rng_normal(Rng *rng) {
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

static int make_output_dir(void) {
#ifdef _WIN32
  if (_mkdir(OUT_DIR) == 0 || errno == EEXIST) return 1;
#else
  if (mkdir(OUT_DIR, 0777) == 0 || errno == EEXIST) return 1;
#endif
  return 0;
}

static int allocate_field(Field *U, int ncell) {
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

static void free_field(Field *U) {
  free(U->h);
  free(U->hu);
  free(U->hv);
  U->h = U->hu = U->hv = NULL;
}

static void copy_field(Field *dst, const Field *src, int ncell) {
  memcpy(dst->h, src->h, (size_t)ncell * sizeof(double));
  memcpy(dst->hu, src->hu, (size_t)ncell * sizeof(double));
  memcpy(dst->hv, src->hv, (size_t)ncell * sizeof(double));
}

static int apply_physical_floor(Field *U, int ncell) {
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

static void initialize_gaussian_bump(Field *U, const Grid *grid) {
  double x0 = 0.5 * grid->lx;
  double y0 = 0.5 * grid->ly;

  for (int j = 0; j < grid->ny; ++j) {
    for (int i = 0; i < grid->nx; ++i) {
      int k = (i + grid->nx * j);
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
double sigma_m, Rng *rng) {
  for (int k = 0; k < ncell; ++k) {
    U->h[k] += sigma_h * rng_normal(rng);
    U->hu[k] += sigma_m * rng_normal(rng);
    U->hv[k] += sigma_m * rng_normal(rng);
  }
  apply_physical_floor(U, ncell);
}

static State get_state_bc(const Field *U, int i, int j, int nx, int ny) {
  State q;

  if (i < 0) i = 0;
  if (i >= nx) i = nx - 1;
  if (j < 0) j = 0;
  if (j >= ny) j = ny - 1;

  {
    int k = (i + nx * j);
    q.h = U->h[k];
    q.hu = U->hu[k];
    q.hv = U->hv[k];
  }
  return q;
}

// --- Forward model: 2D shallow-water equations, finite-volume discretisation ---
// Physical fluxes F(U) = (hu, hu²/h + ½gh², huv/h)^T
static State flux_x(State q) {
  State f;
  double h = (q.h < HMIN ? HMIN : q.h);
  double u = q.hu / h;
  double v = q.hv / h;

  f.h = q.hu;
  f.hu = q.hu * u + 0.5 * G * h * h;
  f.hv = q.hu * v;
  return f;
}

// Physical flux G(U) = (hv, huv/h, hv²/h + ½gh²)^T
static State flux_y(State q) {
  State g;
  double h = (q.h < HMIN ? HMIN : q.h);
  double u = q.hu / h;
  double v = q.hv / h;

  g.h = q.hv;
  g.hu = q.hv * u;
  g.hv = q.hv * v + 0.5 * G * h * h;
  return g;
}

// Rusanov (local Lax-Friedrichs) flux:
//   f̂_{i+½,j} = ½(F_L+F_R) - ½ s_max (U_R-U_L),  s_max = max(|u_L|+c_L, |u_R|+c_R)
static State rusanov_x(State qL, State qR) {
  State fL = flux_x(qL);
  State fR = flux_x(qR);
  State f;
  double hL = (qL.h < HMIN ? HMIN : qL.h);
  double hR = (qR.h < HMIN ? HMIN : qR.h);
  double uL = qL.hu / hL;
  double uR = qR.hu / hR;
  double smax = (fabs(uL) + sqrt(G * hL) > fabs(uR) + sqrt(G * hR) ? fabs(uL) + sqrt(G * hL) : fabs(uR) + sqrt(G * hR));

  f.h = 0.5 * (fL.h + fR.h) - 0.5 * smax * (qR.h - qL.h);
  f.hu = 0.5 * (fL.hu + fR.hu) - 0.5 * smax * (qR.hu - qL.hu);
  f.hv = 0.5 * (fL.hv + fR.hv) - 0.5 * smax * (qR.hv - qL.hv);
  return f;
}

static State rusanov_y(State qB, State qT) {
  State gB = flux_y(qB);
  State gT = flux_y(qT);
  State g;
  double hB = (qB.h < HMIN ? HMIN : qB.h);
  double hT = (qT.h < HMIN ? HMIN : qT.h);
  double vB = qB.hv / hB;
  double vT = qT.hv / hT;
  double smax = (fabs(vB) + sqrt(G * hB) > fabs(vT) + sqrt(G * hT) ? fabs(vB) + sqrt(G * hB) : fabs(vT) + sqrt(G * hT));

  g.h = 0.5 * (gB.h + gT.h) - 0.5 * smax * (qT.h - qB.h);
  g.hu = 0.5 * (gB.hu + gT.hu) - 0.5 * smax * (qT.hu - qB.hu);
  g.hv = 0.5 * (gB.hv + gT.hv) - 0.5 * smax * (qT.hv - qB.hv);
  return g;
}

// CFL time step:  Δt = CFL · min_{cells} (Δx/(|u|+c), Δy/(|v|+c)),  c = √(gh)
static double compute_dt(const Field *U, const Grid *grid) {
  double dt = 1.0e30;

  for (int j = 0; j < grid->ny; ++j) {
    for (int i = 0; i < grid->nx; ++i) {
      int k = (i + grid->nx * j);
      double h = (U->h[k] < HMIN ? HMIN : U->h[k]);
      double u = U->hu[k] / h;
      double v = U->hv[k] / h;
      double c = sqrt(G * h);
      double sx = fabs(u) + c;
      double sy = fabs(v) + c;

      if (sx > 1.0e-14) dt = (dt < grid->dx / sx ? dt : grid->dx / sx);
      if (sy > 1.0e-14) dt = (dt < grid->dy / sy ? dt : grid->dy / sy);
    }
  }
  return CFL * dt;
}

// Explicit Euler: U^{n+1}_{ij} = U^n_{ij} - Δt/Δx ΔF - Δt/Δy ΔG
static void euler_update(const Field *Uin, Field *Uout,
const Grid *grid, double dt) {
  for (int j = 0; j < grid->ny; ++j) {
    for (int i = 0; i < grid->nx; ++i) {
      int k = (i + grid->nx * j);
      State qij = get_state_bc(Uin, i, j, grid->nx, grid->ny);
      State qim1j = get_state_bc(Uin, i - 1, j, grid->nx, grid->ny);
      State qip1j = get_state_bc(Uin, i + 1, j, grid->nx, grid->ny);
      State qijm1 = get_state_bc(Uin, i, j - 1, grid->nx, grid->ny);
      State qijp1 = get_state_bc(Uin, i, j + 1, grid->nx, grid->ny);

      State FxL = rusanov_x(qim1j, qij);
      State FxR = rusanov_x(qij, qip1j);
      State FyB = rusanov_y(qijm1, qij);
      State FyT = rusanov_y(qij, qijp1);

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

// 2-stage RK2: U₁ = Uⁿ + Δt L(Uⁿ),  U₂ = U₁ + Δt L(U₁),  U^{n+1} = ½Uⁿ + ½U₂
static void step_rk2(Field *U, Field *U1, Field *U2,
const Grid *grid, double dt) {
  int ncell = grid->nx * grid->ny;

  euler_update(U, U1, grid, dt);
  euler_update(U1, U2, grid, dt);

  for (int k = 0; k < ncell; ++k) {
    U->h[k] = 0.5 * U->h[k] + 0.5 * U2->h[k];
    U->hu[k] = 0.5 * U->hu[k] + 0.5 * U2->hu[k];
    U->hv[k] = 0.5 * U->hv[k] + 0.5 * U2->hv[k];
  }

  apply_physical_floor(U, ncell);
}

static int model_run(Field *U, Field *U1, Field *U2,
const Grid *grid, double duration) {
  int nsteps = 0;
  double elapsed = 0.0;

  while (elapsed < duration - 1.0e-14) {
    double dt = compute_dt(U, grid);
    if (!isfinite(dt) || dt <= 0.0) return -1;
    if (elapsed + dt > duration) dt = duration - elapsed;

    step_rk2(U, U1, U2, grid, dt);
    elapsed += dt;
    nsteps++;
  }
  return nsteps;
}

static int advance_ensemble(Field *ens, int Ne, Field *work1, Field *work2,
const Grid *grid, double duration) {
  int total_steps = 0;

  for (int e = 0; e < Ne; ++e) {
    int nsteps = model_run(&ens[e], work1, work2, grid, duration);
    if (nsteps < 0) return -1;
    total_steps += nsteps;
  }
  return total_steps;
}

// --- Diagnostics ---
// Ensemble mean:  x̄ = (1/Ne) Σ_e x^{(e)}
static void ensemble_mean(Field *mean, const Field *ens, int Ne, int ncell) {
  for (int k = 0; k < ncell; ++k) {
    mean->h[k] = 0.0;
    mean->hu[k] = 0.0;
    mean->hv[k] = 0.0;
  }

  for (int e = 0; e < Ne; ++e) {
    for (int k = 0; k < ncell; ++k) {
      mean->h[k] += ens[e].h[k];
      mean->hu[k] += ens[e].hu[k];
      mean->hv[k] += ens[e].hv[k];
    }
  }

  for (int k = 0; k < ncell; ++k) {
    mean->h[k] /= Ne;
    mean->hu[k] /= Ne;
    mean->hv[k] /= Ne;
  }
}

// Domain root-mean-square error on h:  √(Σ_k (h_k − ĥ_k)² / N)
static double compute_rmse(const Field *U, const Field *truth, int ncell) {
  double s = 0.0;

  for (int k = 0; k < ncell; ++k) {
    double d = U->h[k] - truth->h[k];
    s += d * d;
  }
  return sqrt(s / ncell);
}

// Ensemble spread σₕ (sample std-dev RMS over domain):  √(Σ_k var_k / N)
static double compute_spread(const Field *ens, const Field *mean, int Ne, int ncell) {
  double s = 0.0;

  for (int k = 0; k < ncell; ++k) {
    double var = 0.0;

    for (int e = 0; e < Ne; ++e) {
      double d = ens[e].h[k] - mean->h[k];
      var += d * d;
    }
    s += var / (Ne - 1);
  }
  return sqrt(s / ncell);
}

// Observation-innovation RMS:  √(Σ_p (y_p − H_p(x̄))² / m)
static double obs_innovation_rms(const Field *mean, const int *obs_idx,
const double *y, int m) {
  double s = 0.0;

  for (int p = 0; p < m; ++p) {
    double d = y[p] - mean->h[obs_idx[p]];
    s += d * d;
  }
  return sqrt(s / m);
}

static void build_observation_network(const Grid *grid, int *obs_idx, int m) {
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
      obs_idx[p++] = (i + grid->nx * j);
    }
  }
}

static void make_observations(const Field *truth, const int *obs_idx,
int m, double obs_std, double *y, Rng *rng) {
  for (int p = 0; p < m; ++p) {
    y[p] = truth->h[obs_idx[p]] + obs_std * rng_normal(rng);
  }
}

// --- Linear algebra utilities ---
// Dense Gauss–Jordan matrix inverse (used for Pyy+R and Te)
static int invert_matrix(double *A, double *Ainv, int m) {
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
      if (v > amax) {
        amax = v;
        pivot = i;
      }
    }

    if (amax < 1.0e-14) {
      free(B);
      return 0;
    }

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

      {
        double factor = Ainv[i * m + k];
        for (int j = 0; j < m; ++j) {
          Ainv[i * m + j] -= factor * Ainv[k * m + j];
          B[i * m + j] -= factor * B[k * m + j];
        }
      }
    }
  }

  memcpy(Ainv, B, (size_t)m * m * sizeof(double));
  free(B);
  return 1;
}

// Symmetric Jacobi eigendecomposition A = V Λ V^T  (used for T^{-1/2} in EnSRF)
static int symmetric_eigen_jacobi(const double *A, double *V, double *eval, int n) {
  double *B = (double *)malloc((size_t)n * n * sizeof(double));
  if (!B) return 0;

  memcpy(B, A, (size_t)n * n * sizeof(double));
  for (int i = 0; i < n * n; ++i) V[i] = 0.0;
  for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;

  for (int iter = 0; iter < 50 * n * n; ++iter) {
    int p = 0;
    int q = 1;
    double max_off = 0.0;

    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        double aij = fabs(B[i * n + j]);
        if (aij > max_off) {
          max_off = aij;
          p = i;
          q = j;
        }
      }
    }

    if (max_off < 1.0e-12) break;

    {
      double app = B[p * n + p];
      double aqq = B[q * n + q];
      double apq = B[p * n + q];
      double tau = (aqq - app) / (2.0 * apq);
      double t = ((tau >= 0.0) ? 1.0 : -1.0) /
      (fabs(tau) + sqrt(1.0 + tau * tau));
      double c = 1.0 / sqrt(1.0 + t * t);
      double s = t * c;

      for (int k = 0; k < n; ++k) {
        if (k != p && k != q) {
          double bkp = B[k * n + p];
          double bkq = B[k * n + q];
          double new_kp = c * bkp - s * bkq;
          double new_kq = s * bkp + c * bkq;
          B[k * n + p] = new_kp;
          B[p * n + k] = new_kp;
          B[k * n + q] = new_kq;
          B[q * n + k] = new_kq;
        }
      }

      B[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
      B[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
      B[p * n + q] = 0.0;
      B[q * n + p] = 0.0;

      for (int k = 0; k < n; ++k) {
        double vkp = V[k * n + p];
        double vkq = V[k * n + q];
        V[k * n + p] = c * vkp - s * vkq;
        V[k * n + q] = s * vkp + c * vkq;
      }
    }
  }

  for (int i = 0; i < n; ++i) {
    eval[i] = B[i * n + i];
    if (!isfinite(eval[i])) {
      free(B);
      return 0;
    }
  }

  free(B);
  return 1;
}

// ===== Stochastic EnKF analysis (Burgers 1998) =====
// P_xy = (Ne-1)⁻¹ Σ_e (xᵉ−x̄)(yᵉ−ȳ)^T,   P_yy = (Ne-1)⁻¹ Σ_e (yᵉ−ȳ)(yᵉ−ȳ)^T
// K = P_xy (P_yy + R)⁻¹,   R = σ²_obs I
// x^{a,(e)} = x^{f,(e)} + K (y + ε^{(e)} − H x^{f,(e)}),   ε^{(e)} ~ N(0,R)
// Perturbed observations supply KRK^T contribution in expectation.
static int stochastic_enkf_analysis(Field *ens, int Ne, const Grid *grid,
const int *obs_idx, int m,
const double *y, double obs_std,
Rng *rng_pert) {
  int ncell = grid->nx * grid->ny;
  int nstate = 3 * ncell;
  double inv_nm1 = 1.0 / (Ne - 1);
  int clipped = 0;
  double *xmean = (double *)calloc((size_t)nstate, sizeof(double));
  double *ymean = (double *)calloc((size_t)m, sizeof(double));
  double *Y = (double *)malloc((size_t)Ne * m * sizeof(double));
  double *Pxy = (double *)calloc((size_t)nstate * m, sizeof(double));
  double *Pyy = (double *)calloc((size_t)m * m, sizeof(double));
  double *S = (double *)calloc((size_t)m * m, sizeof(double));
  double *Sinv = (double *)calloc((size_t)m * m, sizeof(double));
  double *K = (double *)calloc((size_t)nstate * m, sizeof(double));
  double *innov = (double *)malloc((size_t)m * sizeof(double));

  if (!xmean || !ymean || !Y || !Pxy || !Pyy || !S || !Sinv || !K || !innov) {
    fprintf(stderr, "stochastic EnKF allocation failed\n");
    exit(1);
  }

  for (int e = 0; e < Ne; ++e) {
    for (int k = 0; k < ncell; ++k) {
      xmean[k] += ens[e].h[k];
      xmean[ncell + k] += ens[e].hu[k];
      xmean[2 * ncell + k] += ens[e].hv[k];
    }
  }
  for (int k = 0; k < nstate; ++k) xmean[k] /= Ne;

  for (int e = 0; e < Ne; ++e) {
    for (int p = 0; p < m; ++p) {
      double val = ens[e].h[obs_idx[p]];
      Y[e * m + p] = val;
      ymean[p] += val;
    }
  }
  for (int p = 0; p < m; ++p) ymean[p] /= Ne;

  for (int e = 0; e < Ne; ++e) {
    for (int p = 0; p < m; ++p) {
      double dy = Y[e * m + p] - ymean[p];
      for (int k = 0; k < ncell; ++k) {
        Pxy[k * m + p] += (ens[e].h[k] - xmean[k]) * dy;
        Pxy[(ncell + k) * m + p] += (ens[e].hu[k] - xmean[ncell + k]) * dy;
        Pxy[(2 * ncell + k) * m + p] += (ens[e].hv[k] - xmean[2 * ncell + k]) * dy;
      }
    }
  }

  for (int e = 0; e < Ne; ++e) {
    for (int p = 0; p < m; ++p) {
      double dyp = Y[e * m + p] - ymean[p];
      for (int q = 0; q < m; ++q) {
        double dyq = Y[e * m + q] - ymean[q];
        Pyy[p * m + q] += dyp * dyq;
      }
    }
  }

  for (int k = 0; k < nstate * m; ++k) Pxy[k] *= inv_nm1;
  for (int k = 0; k < m * m; ++k) Pyy[k] *= inv_nm1;

  for (int p = 0; p < m; ++p) {
    for (int q = 0; q < m; ++q) S[p * m + q] = Pyy[p * m + q];
    S[p * m + p] += obs_std * obs_std;
  }

  if (!invert_matrix(S, Sinv, m)) {
    fprintf(stderr, "stochastic EnKF inversion failed\n");
    exit(1);
  }

  for (int k = 0; k < nstate; ++k) {
    for (int q = 0; q < m; ++q) {
      double sum = 0.0;
      for (int p = 0; p < m; ++p) {
        sum += Pxy[k * m + p] * Sinv[p * m + q];
      }
      K[k * m + q] = sum;
    }
  }

  for (int e = 0; e < Ne; ++e) {
    for (int p = 0; p < m; ++p) {
      double ypert = y[p] + obs_std * rng_normal(rng_pert);
      innov[p] = ypert - Y[e * m + p];
    }

    for (int k = 0; k < ncell; ++k) {
      double dh = 0.0;
      double dhu = 0.0;
      double dhv = 0.0;

      for (int p = 0; p < m; ++p) {
        dh += K[k * m + p] * innov[p];
        dhu += K[(ncell + k) * m + p] * innov[p];
        dhv += K[(2 * ncell + k) * m + p] * innov[p];
      }

      ens[e].h[k] += dh;
      ens[e].hu[k] += dhu;
      ens[e].hv[k] += dhv;
    }
    clipped += apply_physical_floor(&ens[e], ncell);
  }

  free(xmean);
  free(ymean);
  free(Y);
  free(Pxy);
  free(Pyy);
  free(S);
  free(Sinv);
  free(K);
  free(innov);

  return clipped;
}

// ===== Deterministic EnSRF analysis (symmetric square-root, Whitaker & Hamill 2002) =====
// Ensemble-subspace Hessian:  T_e = I + Y^T R^{-1} Y  ∈ ℝ^{Ne×Ne}
// Mean shift:  w^a = T_e^{-1} Y^T R^{-1} (y − ȳ^f),   x̄^a = x̄^f + X^f w^a
// Square-root:  T_e = V Λ V^T,  T = V Λ^{-1/2} V^T
// Reconstruction:  x^{a,(e)} = x̄^f + √(Ne-1) X^f [T]_e + X^f w^a
// No observation perturbations needed — spread adjusted deterministically.
static int deterministic_enkf_analysis(Field *ens, int Ne, const Grid *grid,
const int *obs_idx, int m,
const double *y, double obs_std) {
  int ncell = grid->nx * grid->ny;
  int nstate = 3 * ncell;
  double inv_ne = 1.0 / Ne;
  double inv_sqrt_nm1 = 1.0 / sqrt((double)(Ne - 1));
  double sqrt_nm1 = sqrt((double)(Ne - 1));
  double R = obs_std * obs_std;
  int clipped = 0;
  double *xmean = (double *)calloc((size_t)nstate, sizeof(double));
  double *ymean = (double *)calloc((size_t)m, sizeof(double));
  double *A = (double *)malloc((size_t)nstate * Ne * sizeof(double));
  double *Yanom = (double *)malloc((size_t)Ne * m * sizeof(double));
  double *T = (double *)calloc((size_t)Ne * Ne, sizeof(double));
  double *Tinv = (double *)calloc((size_t)Ne * Ne, sizeof(double));
  double *Teigvec = (double *)calloc((size_t)Ne * Ne, sizeof(double));
  double *Teigval = (double *)malloc((size_t)Ne * sizeof(double));
  double *Tsqrtinv = (double *)calloc((size_t)Ne * Ne, sizeof(double));
  double *rhs = (double *)calloc((size_t)Ne, sizeof(double));
  double *wa = (double *)calloc((size_t)Ne, sizeof(double));

  if (!xmean || !ymean || !A || !Yanom || !T || !Tinv || !Teigvec ||
  !Teigval || !Tsqrtinv || !rhs || !wa) {
    fprintf(stderr, "deterministic EnKF allocation failed\n");
    exit(1);
  }

  for (int e = 0; e < Ne; ++e) {
    for (int k = 0; k < ncell; ++k) {
      xmean[k] += ens[e].h[k];
      xmean[ncell + k] += ens[e].hu[k];
      xmean[2 * ncell + k] += ens[e].hv[k];
    }
  }
  for (int k = 0; k < nstate; ++k) xmean[k] *= inv_ne;

  for (int e = 0; e < Ne; ++e) {
    for (int p = 0; p < m; ++p) {
      double val = ens[e].h[obs_idx[p]];
      ymean[p] += val;
    }
  }
  for (int p = 0; p < m; ++p) ymean[p] *= inv_ne;

  for (int e = 0; e < Ne; ++e) {
    for (int k = 0; k < ncell; ++k) {
      A[k * Ne + e] = ens[e].h[k] - xmean[k];
      A[(ncell + k) * Ne + e] = ens[e].hu[k] - xmean[ncell + k];
      A[(2 * ncell + k) * Ne + e] = ens[e].hv[k] - xmean[2 * ncell + k];
    }
    for (int p = 0; p < m; ++p) {
      Yanom[e * m + p] = (ens[e].h[obs_idx[p]] - ymean[p]) * inv_sqrt_nm1;
    }
  }

  for (int i = 0; i < Ne; ++i) {
    T[i * Ne + i] = 1.0;
    for (int j = i; j < Ne; ++j) {
      double sum = 0.0;
      for (int p = 0; p < m; ++p) {
        sum += Yanom[i * m + p] * Yanom[j * m + p];
      }
      sum /= R;
      T[i * Ne + j] += sum;
      if (j != i) T[j * Ne + i] += sum;
    }
  }

  if (!invert_matrix(T, Tinv, Ne)) {
    fprintf(stderr, "deterministic EnKF inversion failed\n");
    exit(1);
  }

  for (int i = 0; i < Ne; ++i) {
    double sum = 0.0;
    for (int p = 0; p < m; ++p) {
      sum += Yanom[i * m + p] * (y[p] - ymean[p]);
    }
    rhs[i] = sum / R;
  }
  for (int i = 0; i < Ne; ++i) {
    double sum = 0.0;
    for (int j = 0; j < Ne; ++j) sum += Tinv[i * Ne + j] * rhs[j];
    wa[i] = sum;
  }

  if (!symmetric_eigen_jacobi(T, Teigvec, Teigval, Ne)) {
    fprintf(stderr, "deterministic EnKF eigendecomposition failed\n");
    exit(1);
  }
  for (int i = 0; i < Ne; ++i) {
    if (Teigval[i] < 1.0e-14) Teigval[i] = 1.0e-14;
  }

  for (int i = 0; i < Ne; ++i) {
    for (int j = 0; j < Ne; ++j) {
      double sum = 0.0;
      for (int l = 0; l < Ne; ++l) {
        sum += Teigvec[i * Ne + l] *
        (1.0 / sqrt(Teigval[l])) *
        Teigvec[j * Ne + l];
      }
      Tsqrtinv[i * Ne + j] = sum;
    }
  }

  for (int e = 0; e < Ne; ++e) {
    for (int k = 0; k < ncell; ++k) {
      double new_h = xmean[k];
      double new_hu = xmean[ncell + k];
      double new_hv = xmean[2 * ncell + k];

      for (int j = 0; j < Ne; ++j) {
        double coeff = Tsqrtinv[j * Ne + e] + wa[j] / sqrt_nm1;
        new_h += A[k * Ne + j] * coeff;
        new_hu += A[(ncell + k) * Ne + j] * coeff;
        new_hv += A[(2 * ncell + k) * Ne + j] * coeff;
      }

      ens[e].h[k] = new_h;
      ens[e].hu[k] = new_hu;
      ens[e].hv[k] = new_hv;
    }
    clipped += apply_physical_floor(&ens[e], ncell);
  }

  free(xmean);
  free(ymean);
  free(A);
  free(Yanom);
  free(T);
  free(Tinv);
  free(Teigvec);
  free(Teigval);
  free(Tsqrtinv);
  free(rhs);
  free(wa);
  return clipped;
}

static void write_field_csv(const char *filename, const Field *U,
const Grid *grid, double time) {
  FILE *fp = fopen(filename, "w");
  if (!fp) {
    fprintf(stderr, "warning: cannot open %s\n", filename);
    return;
  }

  fprintf(fp, "# time=%.10f\n", time);
  fprintf(fp, "i,j,x,y,h,hu,hv\n");

  for (int j = 0; j < grid->ny; ++j) {
    for (int i = 0; i < grid->nx; ++i) {
      int k = (i + grid->nx * j);
      double x = (i + 0.5) * grid->dx;
      double y = (j + 0.5) * grid->dy;
      fprintf(fp, "%d,%d,%.10f,%.10f,%.10f,%.10f,%.10f\n",
      i, j, x, y, U->h[k], U->hu[k], U->hv[k]);
    }
  }

  fclose(fp);
}

static void write_cycle_fields(int cycle, const Grid *grid, double t,
const Field *truth,
const Field *mean_stochastic_enkf,
const Field *mean_deterministic_enkf) {
  char path[256];

  snprintf(path, sizeof(path), OUT_DIR "/truth_cycle%03d.csv", cycle);
  write_field_csv(path, truth, grid, t);

  snprintf(path, sizeof(path), OUT_DIR "/stochastic_enkf_mean_cycle%03d.csv", cycle);
  write_field_csv(path, mean_stochastic_enkf, grid, t);

  snprintf(path, sizeof(path), OUT_DIR "/deterministic_enkf_mean_cycle%03d.csv", cycle);
  write_field_csv(path, mean_deterministic_enkf, grid, t);
}

static void write_obs_config(const int *obs_idx, int m) {
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

// ===== Twin experiment main driver =====
// 1. Initialise truth (Gaussian bump) and observation network
// 2. Generate perturbed initial ensemble (same for both filters)
// 3. Cycling loop: forecast → observe → analysis → diagnostics
//    Both filters share the same forecast and observation vector y.
int main(void) {
  Grid grid;
  int ncell = NX * NY;
  int obs_idx[NOBS];
  double y[NOBS];
  Rng rng_init, rng_obs, rng_stochastic_enkf_pert;
  Field truth, work1, work2, mean_stochastic_enkf, mean_deterministic_enkf;
  Field *stochastic_enkf = NULL;
  Field *deterministic_enkf = NULL;

  rng_seed(&rng_init, 2026050801ULL);
  rng_seed(&rng_obs, 2026050802ULL);
  rng_seed(&rng_stochastic_enkf_pert, 2026050803ULL);

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

  stochastic_enkf = (Field *)malloc((size_t)NE * sizeof(Field));
  deterministic_enkf = (Field *)malloc((size_t)NE * sizeof(Field));
  if (!stochastic_enkf || !deterministic_enkf) {
    fprintf(stderr, "top-level allocation failed\n");
    free(stochastic_enkf);
    free(deterministic_enkf);
    return 1;
  }

  if (!allocate_field(&truth, ncell) ||
  !allocate_field(&work1, ncell) ||
  !allocate_field(&work2, ncell) ||
  !allocate_field(&mean_stochastic_enkf, ncell) ||
  !allocate_field(&mean_deterministic_enkf, ncell)) {
    fprintf(stderr, "field allocation failed\n");
    free(stochastic_enkf);
    free(deterministic_enkf);
    return 1;
  }

  for (int e = 0; e < NE; ++e) {
    if (!allocate_field(&stochastic_enkf[e], ncell) ||
    !allocate_field(&deterministic_enkf[e], ncell)) {
      fprintf(stderr, "ensemble allocation failed at member %d\n", e);
      return 1;
    }
  }

  initialize_gaussian_bump(&truth, &grid);
  build_observation_network(&grid, obs_idx, NOBS);
  write_obs_config(obs_idx, NOBS);

  for (int e = 0; e < NE; ++e) {
    copy_field(&stochastic_enkf[e], &truth, ncell);
    perturb_member(&stochastic_enkf[e], ncell, SIGMA_H0, SIGMA_M0, &rng_init);
    copy_field(&deterministic_enkf[e], &stochastic_enkf[e], ncell);
  }

  {
    FILE *metrics = fopen(OUT_DIR "/metrics.csv", "w");
    double t = 0.0;
    if (!metrics) {
      fprintf(stderr, "cannot open metrics.csv\n");
      return 1;
    }

    fprintf(metrics,
    "cycle,time,stochastic_enkf_rmse_f,stochastic_enkf_rmse_a,deterministic_enkf_rmse_f,deterministic_enkf_rmse_a,"
    "stochastic_enkf_spread_f,stochastic_enkf_spread_a,deterministic_enkf_spread_f,deterministic_enkf_spread_a,"
    "stochastic_enkf_innov_rms_f,deterministic_enkf_innov_rms_f,stochastic_enkf_clipped,deterministic_enkf_clipped\n");

    ensemble_mean(&mean_stochastic_enkf, stochastic_enkf, NE, ncell);
    ensemble_mean(&mean_deterministic_enkf, deterministic_enkf, NE, ncell);
    write_cycle_fields(0, &grid, t, &truth, &mean_stochastic_enkf, &mean_deterministic_enkf);

    {
      double init_rmse = compute_rmse(&mean_stochastic_enkf, &truth, ncell);
      double init_spread = compute_spread(stochastic_enkf, &mean_stochastic_enkf, NE, ncell);

      fprintf(metrics,
      "0,%.10f,%.10e,%.10e,%.10e,%.10e,"
      "%.10e,%.10e,%.10e,%.10e,0,0,0,0\n",
      t, init_rmse, init_rmse, init_rmse, init_rmse,
      init_spread, init_spread, init_spread, init_spread);
    }

    printf("Twin comparison: stochastic EnKF vs deterministic EnSRF\n");
    printf("grid=%dx%d Ne=%d obs=%d cycles=%d window=%.4f obs_std=%.4f\n",
    NX, NY, NE, NOBS, NCYCLES, ASSIM_WINDOW, OBS_STD);
    printf("output directory: %s\n\n", OUT_DIR);
    printf("cycle time     EnKF_f     EnKF_a     EnSRF_f    EnSRF_a   clips(E/S)\n");

    for (int cycle = 1; cycle <= NCYCLES; ++cycle) {
      int truth_steps = model_run(&truth, &work1, &work2, &grid, ASSIM_WINDOW);
      int stochastic_enkf_steps = advance_ensemble(stochastic_enkf, NE, &work1, &work2, &grid, ASSIM_WINDOW);
      int deterministic_enkf_steps = advance_ensemble(deterministic_enkf, NE, &work1, &work2, &grid, ASSIM_WINDOW);
      double stochastic_enkf_rmse_f;
      double deterministic_enkf_rmse_f;
      double stochastic_enkf_spread_f;
      double deterministic_enkf_spread_f;
      double stochastic_enkf_innov;
      double deterministic_enkf_innov;
      int stochastic_enkf_clipped;
      int deterministic_enkf_clipped;
      double stochastic_enkf_rmse_a;
      double deterministic_enkf_rmse_a;
      double stochastic_enkf_spread_a;
      double deterministic_enkf_spread_a;

      (void)truth_steps;
      (void)stochastic_enkf_steps;
      (void)deterministic_enkf_steps;
      t += ASSIM_WINDOW;

      ensemble_mean(&mean_stochastic_enkf, stochastic_enkf, NE, ncell);
      ensemble_mean(&mean_deterministic_enkf, deterministic_enkf, NE, ncell);

      stochastic_enkf_rmse_f = compute_rmse(&mean_stochastic_enkf, &truth, ncell);
      deterministic_enkf_rmse_f = compute_rmse(&mean_deterministic_enkf, &truth, ncell);
      stochastic_enkf_spread_f = compute_spread(stochastic_enkf, &mean_stochastic_enkf, NE, ncell);
      deterministic_enkf_spread_f = compute_spread(deterministic_enkf, &mean_deterministic_enkf, NE, ncell);

      make_observations(&truth, obs_idx, NOBS, OBS_STD, y, &rng_obs);
      stochastic_enkf_innov = obs_innovation_rms(&mean_stochastic_enkf, obs_idx, y, NOBS);
      deterministic_enkf_innov = obs_innovation_rms(&mean_deterministic_enkf, obs_idx, y, NOBS);

      stochastic_enkf_clipped = stochastic_enkf_analysis(stochastic_enkf, NE, &grid, obs_idx,
      NOBS, y, OBS_STD,
      &rng_stochastic_enkf_pert);
      deterministic_enkf_clipped = deterministic_enkf_analysis(deterministic_enkf, NE, &grid, obs_idx,
      NOBS, y, OBS_STD);

      ensemble_mean(&mean_stochastic_enkf, stochastic_enkf, NE, ncell);
      ensemble_mean(&mean_deterministic_enkf, deterministic_enkf, NE, ncell);

      stochastic_enkf_rmse_a = compute_rmse(&mean_stochastic_enkf, &truth, ncell);
      deterministic_enkf_rmse_a = compute_rmse(&mean_deterministic_enkf, &truth, ncell);
      stochastic_enkf_spread_a = compute_spread(stochastic_enkf, &mean_stochastic_enkf, NE, ncell);
      deterministic_enkf_spread_a = compute_spread(deterministic_enkf, &mean_deterministic_enkf, NE, ncell);

      fprintf(metrics,
      "%d,%.10f,%.10e,%.10e,%.10e,%.10e,"
      "%.10e,%.10e,%.10e,%.10e,"
      "%.10e,%.10e,%d,%d\n",
      cycle, t,
      stochastic_enkf_rmse_f, stochastic_enkf_rmse_a,
      deterministic_enkf_rmse_f, deterministic_enkf_rmse_a,
      stochastic_enkf_spread_f, stochastic_enkf_spread_a,
      deterministic_enkf_spread_f, deterministic_enkf_spread_a,
      stochastic_enkf_innov, deterministic_enkf_innov,
      stochastic_enkf_clipped, deterministic_enkf_clipped);

      write_cycle_fields(cycle, &grid, t, &truth, &mean_stochastic_enkf, &mean_deterministic_enkf);

      printf("%5d %.3f  %.4e  %.4e  %.4e  %.4e  %d/%d\n",
      cycle, t, stochastic_enkf_rmse_f, stochastic_enkf_rmse_a,
      deterministic_enkf_rmse_f, deterministic_enkf_rmse_a,
      stochastic_enkf_clipped, deterministic_enkf_clipped);

      if (!isfinite(stochastic_enkf_rmse_a) || !isfinite(deterministic_enkf_rmse_a)) {
        fprintf(stderr, "non-finite RMSE at cycle %d\n", cycle);
        break;
      }
    }

    fclose(metrics);
  }

  for (int e = 0; e < NE; ++e) {
    free_field(&stochastic_enkf[e]);
    free_field(&deterministic_enkf[e]);
  }
  free(stochastic_enkf);
  free(deterministic_enkf);
  free_field(&truth);
  free_field(&work1);
  free_field(&work2);
  free_field(&mean_stochastic_enkf);
  free_field(&mean_deterministic_enkf);

  printf("\nDone. Main diagnostics: %s/metrics.csv\n", OUT_DIR);
  return 0;
}
