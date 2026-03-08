#include "swe.h"

#define idx(i,j,nx) ((i) + (nx)*(j))

void swe_step_forward(swe_t *swe, int it);
float compute_cost(swe_t *swe, float *x, float *g);

static void init_gaussian(swe_t *swe, float shift_x_grid, float shift_y_grid, float delta) {
  int i, j;
  float xc = (swe->nx - 1) * swe->dx / 2.0f + shift_x_grid * swe->dx;
  float yc = (swe->ny - 1) * swe->dy / 2.0f + shift_y_grid * swe->dy;
  float sigma = swe->nx * swe->dx / delta;
  for(j = 0; j < swe->ny; j++) {
    for(i = 0; i < swe->nx; i++) {
      float dx = i * swe->dx - xc;
      float dy = j * swe->dy - yc;
      swe->z[idx(i,j,swe->nx)] = expf(-(dx * dx + dy * dy) / (2.0f * sigma * sigma));
      swe->h[idx(i,j,swe->nx)] = 10.0f;
      swe->u[idx(i,j,swe->nx)] = 0.0f;
      swe->v[idx(i,j,swe->nx)] = 0.0f;
    }
  }
}

static int sampled_np(int nx, int ny, int sx, int sy) {
  return ((nx - 1) / sx + 1) * ((ny - 1) / sy + 1);
}

static void setup_problem(swe_t *swe) {
  int it, i, j, k;

  memset(swe, 0, sizeof(*swe));
  swe->nt = 20;
  swe->nx = 41;
  swe->ny = 41;
  swe->dt = 0.1f;
  swe->dx = 10.0f;
  swe->dy = 10.0f;
  swe->gamma = 0.0f;
  swe->f = 0.0f;
  swe->stride_x = 4;
  swe->stride_y = 4;
  swe->n = swe->nx * swe->ny;
  swe->np = sampled_np(swe->nx, swe->ny, swe->stride_x, swe->stride_y);

  swe->h = (float *)calloc(swe->n, sizeof(float));
  swe->H = (float *)calloc(swe->n, sizeof(float));
  swe->z = (float *)calloc(swe->n, sizeof(float));
  swe->u = (float *)calloc(swe->n, sizeof(float));
  swe->v = (float *)calloc(swe->n, sizeof(float));
  swe->a_z = (float *)calloc(swe->n, sizeof(float));
  swe->a_u = (float *)calloc(swe->n, sizeof(float));
  swe->a_v = (float *)calloc(swe->n, sizeof(float));
  swe->a_H = (float *)calloc(swe->n, sizeof(float));
  swe->z_store = (float *)calloc((size_t)swe->nt * swe->n, sizeof(float));
  swe->u_store = (float *)calloc((size_t)swe->nt * swe->n, sizeof(float));
  swe->v_store = (float *)calloc((size_t)swe->nt * swe->n, sizeof(float));
  swe->z_obs = (float *)calloc((size_t)swe->nt * swe->np, sizeof(float));
  swe->z_cal = (float *)calloc((size_t)swe->nt * swe->np, sizeof(float));
  swe->z_res = (float *)calloc((size_t)swe->nt * swe->np, sizeof(float));
  swe->z_temp = (float *)calloc(swe->n, sizeof(float));
  swe->u_temp = (float *)calloc(swe->n, sizeof(float));
  swe->v_temp = (float *)calloc(swe->n, sizeof(float));

  init_gaussian(swe, 0.0f, 0.0f, 15.0f);
  for(it = 0; it < swe->nt; it++) {
    k = 0;
    for(j = 0; j < swe->ny; j++) {
      for(i = 0; i < swe->nx; i++) {
        if(i % swe->stride_x == 0 && j % swe->stride_y == 0) {
          swe->z_obs[k + it * swe->np] = swe->z[idx(i,j,swe->nx)];
          k++;
        }
      }
    }
    swe_step_forward(swe, it);
  }
}

static void free_problem(swe_t *swe) {
  free(swe->h);
  free(swe->H);
  free(swe->z);
  free(swe->u);
  free(swe->v);
  free(swe->a_z);
  free(swe->a_u);
  free(swe->a_v);
  free(swe->a_H);
  free(swe->z_store);
  free(swe->u_store);
  free(swe->v_store);
  free(swe->z_obs);
  free(swe->z_cal);
  free(swe->z_res);
  free(swe->z_temp);
  free(swe->u_temp);
  free(swe->v_temp);
}

static float run_gradcheck(swe_t *swe, float *x, float *g, float *p, float *xp, float *xm) {
  int t, i;
  const int ntrial = 8;
  const float eps = 1e-3f;
  float mean_rel = 0.0f;

  printf("adjoint=swe_step_adjoint\n");

  for(t = 0; t < ntrial; t++) {
    float normp = 0.0f;
    for(i = 0; i < swe->n; i++) {
      p[i] = sinf((0.011f + 0.001f * t) * (float)(i + 3))
           + 0.4f * cosf((0.017f + 0.002f * t) * (float)(i + 7));
      normp += p[i] * p[i];
    }
    normp = sqrtf(normp);
    for(i = 0; i < swe->n; i++) p[i] /= normp;

    {
      float f0 = compute_cost(swe, x, g);
      float gadj = 0.0f;
      float fp, fm, gfd, rel;

      for(i = 0; i < swe->n; i++) {
        xp[i] = x[i] + eps * p[i];
        xm[i] = x[i] - eps * p[i];
        gadj += g[i] * p[i];
      }

      fp = compute_cost(swe, xp, g);
      fm = compute_cost(swe, xm, g);
      gfd = (fp - fm) / (2.0f * eps);
      rel = fabsf(gfd - gadj) / fmaxf(1e-12f, fabsf(gfd));
      mean_rel += rel;

      printf("  trial=%d  f0=%.6e  fd=%.6e  adj=%.6e  rel=%.6e\n",
             t, f0, gfd, gadj, rel);
    }
  }

  mean_rel /= (float)ntrial;
  printf("  mean_rel=%.6e\n", mean_rel);
  return mean_rel;
}

int main(void) {
  swe_t swe;
  float *x, *g, *p, *xp, *xm;
  float mean;

  setup_problem(&swe);

  x = (float *)calloc(swe.n, sizeof(float));
  g = (float *)calloc(swe.n, sizeof(float));
  p = (float *)calloc(swe.n, sizeof(float));
  xp = (float *)calloc(swe.n, sizeof(float));
  xm = (float *)calloc(swe.n, sizeof(float));

  init_gaussian(&swe, 2.5f, -1.5f, 10.0f);
  memcpy(x, swe.z, swe.n * sizeof(float));

  mean = run_gradcheck(&swe, x, g, p, xp, xm);
  printf("summary: mean_rel=%.6e\n", mean);

  free(x);
  free(g);
  free(p);
  free(xp);
  free(xm);
  free_problem(&swe);
  return 0;
}
