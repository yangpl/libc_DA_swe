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

static State rusanov_x(State qL, State qR)
{
    State fL = flux_x(qL);
    State fR = flux_x(qR);
    State f;
    double hL = safe_h(qL.h);
    double hR = safe_h(qR.h);
    double uL = qL.hu / hL;
    double uR = qR.hu / hR;
    double smax = max2(fabs(uL) + sqrt(G * hL), fabs(uR) + sqrt(G * hR));

    f.h = 0.5 * (fL.h + fR.h) - 0.5 * smax * (qR.h - qL.h);
    f.hu = 0.5 * (fL.hu + fR.hu) - 0.5 * smax * (qR.hu - qL.hu);
    f.hv = 0.5 * (fL.hv + fR.hv) - 0.5 * smax * (qR.hv - qL.hv);
    return f;
}

static State rusanov_y(State qB, State qT)
{
    State gB = flux_y(qB);
    State gT = flux_y(qT);
    State g;
    double hB = safe_h(qB.h);
    double hT = safe_h(qT.h);
    double vB = qB.hv / hB;
    double vT = qT.hv / hT;
    double smax = max2(fabs(vB) + sqrt(G * hB), fabs(vT) + sqrt(G * hT));

    g.h = 0.5 * (gB.h + gT.h) - 0.5 * smax * (qT.h - qB.h);
    g.hu = 0.5 * (gB.hu + gT.hu) - 0.5 * smax * (qT.hu - qB.hu);
    g.hv = 0.5 * (gB.hv + gT.hv) - 0.5 * smax * (qT.hv - qB.hv);
    return g;
}

static double compute_dt(const Field *U, const Grid *grid)
{
    double dt = 1.0e30;

    for (int j = 0; j < grid->ny; ++j) {
        for (int i = 0; i < grid->nx; ++i) {
            int k = idx2(i, j, grid->nx);
            double h = safe_h(U->h[k]);
            double u = U->hu[k] / h;
            double v = U->hv[k] / h;
            double c = sqrt(G * h);
            double sx = fabs(u) + c;
            double sy = fabs(v) + c;

            if (sx > 1.0e-14) dt = min2(dt, grid->dx / sx);
            if (sy > 1.0e-14) dt = min2(dt, grid->dy / sy);
        }
    }
    return CFL * dt;
}

static void euler_update(const Field *Uin, Field *Uout,
                         const Grid *grid, double dt)
{
    for (int j = 0; j < grid->ny; ++j) {
        for (int i = 0; i < grid->nx; ++i) {
            int k = idx2(i, j, grid->nx);
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

static void step_rk2(Field *U, Field *U1, Field *U2,
                     const Grid *grid, double dt)
{
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

static int model_run_for(Field *U, Field *U1, Field *U2,
                         const Grid *grid, double duration)
{
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

static int advance_ensemble_for(Field *ens, int Ne, Field *work1, Field *work2,
                                const Grid *grid, double duration)
{
    int total_steps = 0;

    for (int e = 0; e < Ne; ++e) {
        int nsteps = model_run_for(&ens[e], work1, work2, grid, duration);
        if (nsteps < 0) return -1;
        total_steps += nsteps;
    }
    return total_steps;
}

static void ensemble_mean(Field *mean, const Field *ens, int Ne, int ncell)
{
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

static double rmse_h(const Field *U, const Field *truth, int ncell)
{
    double s = 0.0;

    for (int k = 0; k < ncell; ++k) {
        double d = U->h[k] - truth->h[k];
        s += d * d;
    }
    return sqrt(s / ncell);
}

static double spread_h(const Field *ens, const Field *mean, int Ne, int ncell)
{
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

static double obs_innovation_rms(const Field *mean, const int *obs_idx,
                                 const double *y, int m)
{
    double s = 0.0;

    for (int p = 0; p < m; ++p) {
        double d = y[p] - mean->h[obs_idx[p]];
        s += d * d;
    }
    return sqrt(s / m);
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

static int invert_matrix(double *A, double *Ainv, int m)
{
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

static int stochastic_enkf_analysis(Field *ens, int Ne, const Grid *grid,
                                    const int *obs_idx, int m,
                                    const double *y, double obs_std,
                                    Rng *rng_pert)
{
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

static int ensrf_analysis(Field *ens, int Ne, const Grid *grid,
                          const int *obs_idx, int m,
                          const double *y, double obs_std)
{
    int ncell = grid->nx * grid->ny;
    double inv_nm1 = 1.0 / (Ne - 1);
    double R = obs_std * obs_std;
    int clipped = 0;
    double *hxp = (double *)malloc((size_t)Ne * sizeof(double));

    (void)grid;

    if (!hxp) {
        fprintf(stderr, "EnSRF allocation failed\n");
        exit(1);
    }

    for (int p = 0; p < m; ++p) {
        int ok = obs_idx[p];
        double hxbar = 0.0;
        double hpht = 0.0;
        double denom;
        double innov;
        double alpha;

        for (int e = 0; e < Ne; ++e) hxbar += ens[e].h[ok];
        hxbar /= Ne;

        for (int e = 0; e < Ne; ++e) {
            hxp[e] = ens[e].h[ok] - hxbar;
            hpht += hxp[e] * hxp[e];
        }
        hpht *= inv_nm1;

        denom = hpht + R;
        if (denom < 1.0e-30) continue;

        innov = y[p] - hxbar;
        alpha = (R <= 0.0) ? 1.0 : 1.0 / (1.0 + sqrt(R / denom));

        for (int k = 0; k < ncell; ++k) {
            double mean_h = 0.0;
            double mean_hu = 0.0;
            double mean_hv = 0.0;
            double cov_h = 0.0;
            double cov_hu = 0.0;
            double cov_hv = 0.0;
            double kh;
            double khu;
            double khv;

            for (int e = 0; e < Ne; ++e) {
                mean_h += ens[e].h[k];
                mean_hu += ens[e].hu[k];
                mean_hv += ens[e].hv[k];
            }
            mean_h /= Ne;
            mean_hu /= Ne;
            mean_hv /= Ne;

            for (int e = 0; e < Ne; ++e) {
                cov_h += (ens[e].h[k] - mean_h) * hxp[e];
                cov_hu += (ens[e].hu[k] - mean_hu) * hxp[e];
                cov_hv += (ens[e].hv[k] - mean_hv) * hxp[e];
            }
            cov_h *= inv_nm1;
            cov_hu *= inv_nm1;
            cov_hv *= inv_nm1;

            kh = cov_h / denom;
            khu = cov_hu / denom;
            khv = cov_hv / denom;

            for (int e = 0; e < Ne; ++e) {
                ens[e].h[k] += kh * innov - alpha * kh * hxp[e];
                ens[e].hu[k] += khu * innov - alpha * khu * hxp[e];
                ens[e].hv[k] += khv * innov - alpha * khv * hxp[e];
            }
        }

        for (int e = 0; e < Ne; ++e) {
            clipped += apply_physical_floor(&ens[e], ncell);
        }
    }

    free(hxp);
    return clipped;
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

static void write_cycle_fields(int cycle, const Grid *grid, double t,
                               const Field *truth,
                               const Field *mean_enkf,
                               const Field *mean_ensrf)
{
    char path[256];

    snprintf(path, sizeof(path), OUT_DIR "/truth_cycle%03d.csv", cycle);
    write_field_csv(path, truth, grid, t);

    snprintf(path, sizeof(path), OUT_DIR "/stoch_enkf_mean_cycle%03d.csv", cycle);
    write_field_csv(path, mean_enkf, grid, t);

    snprintf(path, sizeof(path), OUT_DIR "/ensrf_mean_cycle%03d.csv", cycle);
    write_field_csv(path, mean_ensrf, grid, t);
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

int main(void)
{
    Grid grid;
    int ncell = NX * NY;
    int obs_idx[NOBS];
    double y[NOBS];
    Rng rng_init, rng_obs, rng_enkf_pert;
    Field truth, work1, work2, mean_enkf, mean_ensrf;
    Field *enkf = NULL;
    Field *ensrf = NULL;

    rng_seed(&rng_init, 2026050801ULL);
    rng_seed(&rng_obs, 2026050802ULL);
    rng_seed(&rng_enkf_pert, 2026050803ULL);

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

    enkf = (Field *)malloc((size_t)NE * sizeof(Field));
    ensrf = (Field *)malloc((size_t)NE * sizeof(Field));
    if (!enkf || !ensrf) {
        fprintf(stderr, "top-level allocation failed\n");
        free(enkf);
        free(ensrf);
        return 1;
    }

    if (!allocate_field(&truth, ncell) ||
        !allocate_field(&work1, ncell) ||
        !allocate_field(&work2, ncell) ||
        !allocate_field(&mean_enkf, ncell) ||
        !allocate_field(&mean_ensrf, ncell)) {
        fprintf(stderr, "field allocation failed\n");
        free(enkf);
        free(ensrf);
        return 1;
    }

    for (int e = 0; e < NE; ++e) {
        if (!allocate_field(&enkf[e], ncell) ||
            !allocate_field(&ensrf[e], ncell)) {
            fprintf(stderr, "ensemble allocation failed at member %d\n", e);
            return 1;
        }
    }

    initialize_gaussian_bump(&truth, &grid);
    build_observation_network(&grid, obs_idx, NOBS);
    write_obs_config(obs_idx, NOBS);

    for (int e = 0; e < NE; ++e) {
        copy_field(&enkf[e], &truth, ncell);
        perturb_member(&enkf[e], ncell, SIGMA_H0, SIGMA_M0, &rng_init);
        copy_field(&ensrf[e], &enkf[e], ncell);
    }

    {
        FILE *metrics = fopen(OUT_DIR "/metrics.csv", "w");
        double t = 0.0;
        if (!metrics) {
            fprintf(stderr, "cannot open metrics.csv\n");
            return 1;
        }

        fprintf(metrics,
                "cycle,time,enkf_rmse_f,enkf_rmse_a,ensrf_rmse_f,ensrf_rmse_a,"
                "enkf_spread_f,enkf_spread_a,ensrf_spread_f,ensrf_spread_a,"
                "enkf_innov_rms_f,ensrf_innov_rms_f,enkf_clipped,ensrf_clipped\n");

        ensemble_mean(&mean_enkf, enkf, NE, ncell);
        ensemble_mean(&mean_ensrf, ensrf, NE, ncell);
        write_cycle_fields(0, &grid, t, &truth, &mean_enkf, &mean_ensrf);

        {
            double init_rmse = rmse_h(&mean_enkf, &truth, ncell);
            double init_spread = spread_h(enkf, &mean_enkf, NE, ncell);

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
            int truth_steps = model_run_for(&truth, &work1, &work2, &grid, ASSIM_WINDOW);
            int enkf_steps = advance_ensemble_for(enkf, NE, &work1, &work2, &grid, ASSIM_WINDOW);
            int ensrf_steps = advance_ensemble_for(ensrf, NE, &work1, &work2, &grid, ASSIM_WINDOW);
            double enkf_rmse_f;
            double ensrf_rmse_f;
            double enkf_spread_f;
            double ensrf_spread_f;
            double enkf_innov;
            double ensrf_innov;
            int enkf_clipped;
            int ensrf_clipped;
            double enkf_rmse_a;
            double ensrf_rmse_a;
            double enkf_spread_a;
            double ensrf_spread_a;

            (void)truth_steps;
            (void)enkf_steps;
            (void)ensrf_steps;
            t += ASSIM_WINDOW;

            ensemble_mean(&mean_enkf, enkf, NE, ncell);
            ensemble_mean(&mean_ensrf, ensrf, NE, ncell);

            enkf_rmse_f = rmse_h(&mean_enkf, &truth, ncell);
            ensrf_rmse_f = rmse_h(&mean_ensrf, &truth, ncell);
            enkf_spread_f = spread_h(enkf, &mean_enkf, NE, ncell);
            ensrf_spread_f = spread_h(ensrf, &mean_ensrf, NE, ncell);

            make_observations(&truth, obs_idx, NOBS, OBS_STD, y, &rng_obs);
            enkf_innov = obs_innovation_rms(&mean_enkf, obs_idx, y, NOBS);
            ensrf_innov = obs_innovation_rms(&mean_ensrf, obs_idx, y, NOBS);

            enkf_clipped = stochastic_enkf_analysis(enkf, NE, &grid, obs_idx,
                                                    NOBS, y, OBS_STD,
                                                    &rng_enkf_pert);
            ensrf_clipped = ensrf_analysis(ensrf, NE, &grid, obs_idx,
                                           NOBS, y, OBS_STD);

            ensemble_mean(&mean_enkf, enkf, NE, ncell);
            ensemble_mean(&mean_ensrf, ensrf, NE, ncell);

            enkf_rmse_a = rmse_h(&mean_enkf, &truth, ncell);
            ensrf_rmse_a = rmse_h(&mean_ensrf, &truth, ncell);
            enkf_spread_a = spread_h(enkf, &mean_enkf, NE, ncell);
            ensrf_spread_a = spread_h(ensrf, &mean_ensrf, NE, ncell);

            fprintf(metrics,
                    "%d,%.10f,%.10e,%.10e,%.10e,%.10e,"
                    "%.10e,%.10e,%.10e,%.10e,"
                    "%.10e,%.10e,%d,%d\n",
                    cycle, t,
                    enkf_rmse_f, enkf_rmse_a,
                    ensrf_rmse_f, ensrf_rmse_a,
                    enkf_spread_f, enkf_spread_a,
                    ensrf_spread_f, ensrf_spread_a,
                    enkf_innov, ensrf_innov,
                    enkf_clipped, ensrf_clipped);

            write_cycle_fields(cycle, &grid, t, &truth, &mean_enkf, &mean_ensrf);

            printf("%5d %.3f  %.4e  %.4e  %.4e  %.4e  %d/%d\n",
                   cycle, t, enkf_rmse_f, enkf_rmse_a,
                   ensrf_rmse_f, ensrf_rmse_a,
                   enkf_clipped, ensrf_clipped);

            if (!isfinite(enkf_rmse_a) || !isfinite(ensrf_rmse_a)) {
                fprintf(stderr, "non-finite RMSE at cycle %d\n", cycle);
                break;
            }
        }

        fclose(metrics);
    }

    for (int e = 0; e < NE; ++e) {
        free_field(&enkf[e]);
        free_field(&ensrf[e]);
    }
    free(enkf);
    free(ensrf);
    free_field(&truth);
    free_field(&work1);
    free_field(&work2);
    free_field(&mean_enkf);
    free_field(&mean_ensrf);

    printf("\nDone. Main diagnostics: %s/metrics.csv\n", OUT_DIR);
    return 0;
}
