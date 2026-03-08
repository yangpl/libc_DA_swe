#include "swe.h"

#define idx(i,j,nx) ((i) + (nx)*(j))

void swe_step_forward(swe_t *swe, int it);
void swe_step_adjoint(swe_t *swe, int it);

static float frand_uniform(unsigned int *seed) {
  *seed = 1664525u * (*seed) + 1013904223u;
  return ((float)((*seed >> 8) & 0x00FFFFFF) / 16777215.0f) * 2.0f - 1.0f;
}

static void alloc_swe(swe_t *swe, int nx, int ny, int nt) {
  memset(swe, 0, sizeof(*swe));
  swe->nx = nx;
  swe->ny = ny;
  swe->nt = nt;
  swe->dx = 10.0f;
  swe->dy = 10.0f;
  swe->dt = 0.1f;
  swe->gamma = 0.0f;
  swe->f = 0.0f;
  swe->n = nx * ny;
  swe->stride_x = 4;
  swe->stride_y = 4;
  swe->np = ((nx - 1) / swe->stride_x + 1) * ((ny - 1) / swe->stride_y + 1);

  swe->h = (float *)calloc(swe->n, sizeof(float));
  swe->H = (float *)calloc(swe->n, sizeof(float));
  swe->z = (float *)calloc(swe->n, sizeof(float));
  swe->u = (float *)calloc(swe->n, sizeof(float));
  swe->v = (float *)calloc(swe->n, sizeof(float));
  swe->a_z = (float *)calloc(swe->n, sizeof(float));
  swe->a_u = (float *)calloc(swe->n, sizeof(float));
  swe->a_v = (float *)calloc(swe->n, sizeof(float));
  swe->a_H = (float *)calloc(swe->n, sizeof(float));
  swe->z_store = (float *)calloc((size_t)nt * swe->n, sizeof(float));
  swe->u_store = (float *)calloc((size_t)nt * swe->n, sizeof(float));
  swe->v_store = (float *)calloc((size_t)nt * swe->n, sizeof(float));
  swe->z_obs = (float *)calloc((size_t)nt * swe->np, sizeof(float));
  swe->z_cal = (float *)calloc((size_t)nt * swe->np, sizeof(float));
  swe->z_res = (float *)calloc((size_t)nt * swe->np, sizeof(float));
  swe->z_temp = (float *)calloc(swe->n, sizeof(float));
  swe->u_temp = (float *)calloc(swe->n, sizeof(float));
  swe->v_temp = (float *)calloc(swe->n, sizeof(float));
}

static void free_swe(swe_t *swe) {
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

static void copy_h_and_state(const swe_t *src, swe_t *dst) {
  memcpy(dst->h, src->h, src->n * sizeof(float));
  memcpy(dst->z, src->z, src->n * sizeof(float));
  memcpy(dst->u, src->u, src->n * sizeof(float));
  memcpy(dst->v, src->v, src->n * sizeof(float));
}

static void run_forward_nt(swe_t *swe) {
  int it;
  for(it = 0; it < swe->nt; it++) swe_step_forward(swe, it);
}

static float dot3(const float *a1, const float *a2, const float *a3,
                  const float *b1, const float *b2, const float *b3, int n) {
  int i;
  float s = 0.0f;
  for(i = 0; i < n; i++) s += a1[i] * b1[i] + a2[i] * b2[i] + a3[i] * b3[i];
  return s;
}

static void apply_multistep_jacobian_fd(const swe_t *base,
                                        const float *dz0, const float *du0, const float *dv0,
                                        float *JdzN, float *JduN, float *JdvN, float eps) {
  int i;
  swe_t plus, minus;
  alloc_swe(&plus, base->nx, base->ny, base->nt);
  alloc_swe(&minus, base->nx, base->ny, base->nt);
  copy_h_and_state(base, &plus);
  copy_h_and_state(base, &minus);

  for(i = 0; i < base->n; i++) {
    plus.z[i] += eps * dz0[i];
    plus.u[i] += eps * du0[i];
    plus.v[i] += eps * dv0[i];
    minus.z[i] -= eps * dz0[i];
    minus.u[i] -= eps * du0[i];
    minus.v[i] -= eps * dv0[i];
  }

  run_forward_nt(&plus);
  run_forward_nt(&minus);

  for(i = 0; i < base->n; i++) {
    JdzN[i] = (plus.z[i] - minus.z[i]) / (2.0f * eps);
    JduN[i] = (plus.u[i] - minus.u[i]) / (2.0f * eps);
    JdvN[i] = (plus.v[i] - minus.v[i]) / (2.0f * eps);
  }

  free_swe(&plus);
  free_swe(&minus);
}

static float run_adjcheck(const swe_t *base, int ntrial, float eps, unsigned int seed0) {
  int t, i, it;
  float mean_rel = 0.0f;
  float *dz0 = (float *)calloc(base->n, sizeof(float));
  float *du0 = (float *)calloc(base->n, sizeof(float));
  float *dv0 = (float *)calloc(base->n, sizeof(float));
  float *lzN = (float *)calloc(base->n, sizeof(float));
  float *luN = (float *)calloc(base->n, sizeof(float));
  float *lvN = (float *)calloc(base->n, sizeof(float));
  float *JdzN = (float *)calloc(base->n, sizeof(float));
  float *JduN = (float *)calloc(base->n, sizeof(float));
  float *JdvN = (float *)calloc(base->n, sizeof(float));

  printf("adjoint=swe_step_adjoint\n");
  for(t = 0; t < ntrial; t++) {
    unsigned int seed = seed0 + (unsigned int)(101 * t);
    float nrm_d = 0.0f, nrm_l = 0.0f;
    float lhs, rhs, rel;
    swe_t adj;

    for(i = 0; i < base->n; i++) {
      dz0[i] = frand_uniform(&seed);
      du0[i] = frand_uniform(&seed);
      dv0[i] = frand_uniform(&seed);
      lzN[i] = frand_uniform(&seed);
      luN[i] = frand_uniform(&seed);
      lvN[i] = frand_uniform(&seed);
      nrm_d += dz0[i] * dz0[i] + du0[i] * du0[i] + dv0[i] * dv0[i];
      nrm_l += lzN[i] * lzN[i] + luN[i] * luN[i] + lvN[i] * lvN[i];
    }
    nrm_d = sqrtf(nrm_d);
    nrm_l = sqrtf(nrm_l);
    for(i = 0; i < base->n; i++) {
      dz0[i] /= nrm_d; du0[i] /= nrm_d; dv0[i] /= nrm_d;
      lzN[i] /= nrm_l; luN[i] /= nrm_l; lvN[i] /= nrm_l;
    }

    apply_multistep_jacobian_fd(base, dz0, du0, dv0, JdzN, JduN, JdvN, eps);
    lhs = dot3(JdzN, JduN, JdvN, lzN, luN, lvN, base->n);

    alloc_swe(&adj, base->nx, base->ny, base->nt);
    copy_h_and_state(base, &adj);
    run_forward_nt(&adj);
    memcpy(adj.a_z, lzN, base->n * sizeof(float));
    memcpy(adj.a_u, luN, base->n * sizeof(float));
    memcpy(adj.a_v, lvN, base->n * sizeof(float));
    for(it = base->nt - 1; it >= 0; it--) {
      swe_step_adjoint(&adj, it);
    }

    rhs = dot3(dz0, du0, dv0, adj.a_z, adj.a_u, adj.a_v, base->n);
    rel = fabsf(lhs - rhs) / fmaxf(1e-12f, fmaxf(fabsf(lhs), fabsf(rhs)));
    mean_rel += rel;
    printf("  trial=%d lhs=%.6e rhs=%.6e rel=%.6e\n", t, lhs, rhs, rel);
    free_swe(&adj);
  }

  mean_rel /= (float)ntrial;
  printf("  mean_rel=%.6e\n", mean_rel);

  free(dz0); free(du0); free(dv0);
  free(lzN); free(luN); free(lvN);
  free(JdzN); free(JduN); free(JdvN);
  return mean_rel;
}

int main(int argc, char **argv) {
  int i;
  int nx = 41, ny = 41, nt = 20, ntrial = 8;
  float eps = 1e-4f;
  unsigned int seed = 12345u;
  swe_t base;
  float mean;

  if(argc > 1) nt = atoi(argv[1]);
  if(argc > 2) ntrial = atoi(argv[2]);
  if(argc > 3) eps = (float)atof(argv[3]);

  alloc_swe(&base, nx, ny, nt);
  for(i = 0; i < base.n; i++) {
    base.h[i] = 10.0f;
    base.z[i] = 0.3f * frand_uniform(&seed);
    base.u[i] = 0.2f * frand_uniform(&seed);
    base.v[i] = 0.2f * frand_uniform(&seed);
  }

  printf("multistep adjoint check: nt=%d ntrial=%d eps=%.1e\n", nt, ntrial, eps);
  mean = run_adjcheck(&base, ntrial, eps, 123u);
  printf("summary: mean_rel=%.6e\n", mean);

  free_swe(&base);
  return 0;
}
