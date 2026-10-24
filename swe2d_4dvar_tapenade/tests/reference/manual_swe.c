/* Nonlinear 2-D shallow-water model and exact discrete adjoint.
 * Cell-centred conservative variables (h, hu, hv), Rusanov flux, SSP-RK2.
 * Zero-gradient open boundaries. See README.md for limitations and C API.
 * No state clipping: invalid or CFL-unsafe trajectories are rejected.
 */
#include "swe2d_4dvar.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define G 9.81
#define HMIN 1.e-6

typedef struct { int nx, ny; double lx, ly, dx, dy; } Grid;
typedef struct { double *h, *hu, *hv; } Field;
typedef struct { double h, hu, hv; } State;
static int idx2(int i, int j, int nx) { return i + nx*j; }
static double max2(double a, double b) { return a > b ? a : b; }
static double safe_h(double h) { return h < HMIN ? HMIN : h; }
static double signum(double x) { return (x > 0) - (x < 0); }
static Field field(double *p, int n) { Field f = {p, p+n, p+2*n}; return f; }
static void zero_field(Field *f, int n) { memset(f->h, 0, 3*(size_t)n*sizeof(double)); }
static void field_axpy(Field *y, double a, const Field *x, int n) {
  for (int k=0; k<3*n; ++k) y->h[k] += a*x->h[k];
}
static void flux_jacobian_x(State q, double J[9])
{
  /*
   * Jacobian of the physical x-flux
   *
   *   F(q) = [ hu,
   *            hu^2 / h + 0.5 g h^2,
   *            hu hv / h ].
   *
   * Entries are row-major: J[3*r+c] = dF_r/dq_c for q=(h,hu,hv).
   * The dhs factor freezes the h derivative when h is below the dry floor,
   * matching the safe_h() branch used by the nonlinear flux.
   */
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
  /*
   * Jacobian of the physical y-flux
   *
   *   G(q) = [ hv,
   *            hu hv / h,
   *            hv^2 / h + 0.5 g h^2 ].
   */
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
  /*
   * Local Rusanov signal speed in x: |u| + sqrt(g h), plus its derivative.
   * signum() gives a piecewise derivative of |u|; at u=0 the chosen
   * subgradient is zero, which is adequate for the discrete adjoint check.
   */
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
  /*
   * y-direction counterpart of wavespeed_grad_x(): speed is
   * |v| + sqrt(g h), and grad is d(speed)/d(h,hu,hv).
   */
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
  /*
   * Local Lax-Friedrichs/Rusanov interface flux:
   *
   *   Fhat = 0.5(F_L + F_R) - 0.5 s (q_R - q_L),
   *
   * where s is the larger one-sided gravity-wave speed.
   */
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
  /* Same Rusanov flux as rusanov_x(), applied to a bottom/top y-face. */
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
  /*
   * Derivatives of the numerical x-flux with respect to the left and right
   * interface states. The derivative of max(aL,aR) is assigned to whichever
   * side wins the comparison, consistent with the nonlinear branch.
   */
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
      /*
       * Differentiate 0.5(F_L+F_R)-0.5*s*dq. The +/- identity terms come
       * from d(dq)/dqL=-I and d(dq)/dqR=I; the gs terms come from ds/dq.
       */
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
  /*
   * y-face version of rusanov_x_jacobians(). The bottom/top naming follows
   * the finite-volume stencil in euler_update() and euler_adjoint().
   */
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


/* Compute every interface once. Boundary faces map both states to the
 * boundary cell, and the transpose uses exactly the same mapping. */
static void euler_update(const Field *u, Field *v, const Grid *g, double dt) {
  int nc = g->nx*g->ny;
  memcpy(v->h, u->h, 3*(size_t)nc*sizeof(double));
  for (int axis=0; axis<2; ++axis) {
    int across = axis ? g->ny : g->nx;
    int along = axis ? g->nx : g->ny;
    double scale = dt/(axis ? g->dy : g->dx);
    for (int b=0; b<along; ++b) for (int a=0; a<=across; ++a) {
      int lo = a > 0 ? a-1 : 0, hi = a < across ? a : across-1;
      int l = axis ? idx2(b,lo,g->nx) : idx2(lo,b,g->nx);
      int r = axis ? idx2(b,hi,g->nx) : idx2(hi,b,g->nx);
      State ql = {u->h[l],u->hu[l],u->hv[l]};
      State qr = {u->h[r],u->hu[r],u->hv[r]}, f;
      if (axis) rusanov_y(ql,qr,&f); else rusanov_x(ql,qr,&f);
      if (a>0) { v->h[l]-=scale*f.h; v->hu[l]-=scale*f.hu; v->hv[l]-=scale*f.hv; }
      if (a<across) { v->h[r]+=scale*f.h; v->hu[r]+=scale*f.hu; v->hv[r]+=scale*f.hv; }
    }
  }
}
static void add_matT_vec_to_cell(Field *lam, int cell, const double J[9],
                                 const double v[3], double coeff)
{
  /*
   * Accumulate coeff * J^T v into one cell of an adjoint field. This is the
   * basic reverse-mode operation used for each numerical-flux dependency.
   */
  lam->h[cell]  += coeff * (J[0] * v[0] + J[3] * v[1] + J[6] * v[2]);
  lam->hu[cell] += coeff * (J[1] * v[0] + J[4] * v[1] + J[7] * v[2]);
  lam->hv[cell] += coeff * (J[2] * v[0] + J[5] * v[1] + J[8] * v[2]);
}


static void euler_adjoint(const Field *u, const Field *lv, Field *lu,
                          const Grid *g, double dt) {
  int nc = g->nx*g->ny;
  memcpy(lu->h, lv->h, 3*(size_t)nc*sizeof(double));
  for (int axis=0; axis<2; ++axis) {
    int across = axis ? g->ny : g->nx;
    int along = axis ? g->nx : g->ny;
    double scale = dt/(axis ? g->dy : g->dx);
    for (int b=0; b<along; ++b) for (int a=0; a<=across; ++a) {
      int lo = a > 0 ? a-1 : 0, hi = a < across ? a : across-1;
      int l = axis ? idx2(b,lo,g->nx) : idx2(lo,b,g->nx);
      int r = axis ? idx2(b,hi,g->nx) : idx2(hi,b,g->nx);
      State ql = {u->h[l],u->hu[l],u->hv[l]};
      State qr = {u->h[r],u->hu[r],u->hv[r]};
      double jl[9],jr[9],v[3];
      for (int c=0; c<3; ++c)
        v[c] = (a<across ? lv->h[c*nc+r] : 0) - (a>0 ? lv->h[c*nc+l] : 0);
      if (axis) rusanov_y_jacobians(ql,qr,jl,jr); else rusanov_x_jacobians(ql,qr,jl,jr);
      add_matT_vec_to_cell(lu,l,jl,v,scale);
      add_matT_vec_to_cell(lu,r,jr,v,scale);
    }
  }
}
static void rk2_adjoint_step(const Field *Qn, const Field *Q1,
                             const Field *lam_np1, Field *lam_n,
                             Field *lam_stage2, Field *lam_stage1,
                             const Grid *grid, double dt)
{
  /*
   * Reverse of the RK2 composition used in step_rk2_store().
   *
   * Q_{n+1} = 0.5 Q_n + 0.5 E(Q1), Q1 = E(Q_n).
   * Therefore lam_np1 is split: half goes directly to Q_n, half goes through
   * the second Euler update at Q1, then through the first Euler update at Qn.
   */
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


static int valid_args(int nx,int ny,double lx,double ly,double dt,int ns,int stride) {
  return nx>=2 && ny>=2 && nx<=4096 && ny<=4096 &&
    isfinite(lx) && lx>0 && isfinite(ly) && ly>0 && isfinite(dt) && dt>0 &&
    ns>0 && stride>0 && ns%stride==0;
}
static int check_state(const Field *u,const Grid *g,double dt,double *diag) {
  double maxx=0,maxy=0;
  int n=g->nx*g->ny;
  for(int k=0;k<n;++k) {
    double h=u->h[k], m=u->hu[k], v=u->hv[k];
    if(!isfinite(h)||!isfinite(m)||!isfinite(v)||h<=HMIN) return 3;
    diag[0]=fmin(diag[0],h);
    double c=sqrt(G*h);
    maxx=fmax(maxx,fabs(m/h)+c); maxy=fmax(maxy,fabs(v/h)+c);
  }
  double cfl=dt*(maxx/g->dx+maxy/g->dy);
  diag[1]=fmax(diag[1],cfl);
  return cfl<=0.9 ? 0 : 4;
}
static int step_checked(const Field *u,Field *stage,Field *tmp,Field *out,
                        const Grid *g,double dt,double *diag) {
  int err=check_state(u,g,dt,diag);
  if(err) return err;
  euler_update(u,stage,g,dt);
  if((err=check_state(stage,g,dt,diag))) return err;
  euler_update(stage,tmp,g,dt);
  if((err=check_state(tmp,g,dt,diag))) return err;
  for(int k=0;k<3*g->nx*g->ny;++k) out->h[k]=0.5*(u->h[k]+tmp->h[k]);
  return check_state(out,g,dt,diag);
}
int swe_forward(const double *initial,int nx,int ny,double lx,double ly,
                double dt,int ns,int stride,double *checkpoints,double *diag) {
  if(!initial||!checkpoints||!diag||!valid_args(nx,ny,lx,ly,dt,ns,stride)) return 1;
  int n=nx*ny,err=0;
  size_t nd=3*(size_t)n, bytes=nd*sizeof(double);
  double *buf=calloc(4*nd,sizeof(double));
  if(!buf) return 2;
  Grid g={nx,ny,lx,ly,lx/nx,ly/ny};
  Field u=field(buf,n),stage=field(buf+nd,n),tmp=field(buf+2*nd,n),v=field(buf+3*nd,n);
  memcpy(u.h,initial,bytes); memcpy(checkpoints,initial,bytes);
  diag[0]=INFINITY; diag[1]=0;
  for(int s=1;s<=ns;++s) {
    if((err=step_checked(&u,&stage,&tmp,&v,&g,dt,diag))) break;
    Field swap=u;u=v;v=swap;
    if(s%stride==0) memcpy(checkpoints+(size_t)(s/stride)*nd,u.h,bytes);
  }
  free(buf);return err;
}
int swe_evaluate(const double *initial,int nx,int ny,double lx,double ly,
                 double dt,int ns,int stride,int m,const int *indices,
                 const double *obs,double sigma,double *cost,double *gradient,double *diag) {
  if(!initial||!indices||!obs||!cost||!diag||m<1||!isfinite(sigma)||sigma<=0||
     !valid_args(nx,ny,lx,ly,dt,ns,stride)) return 1;
  int n=nx*ny, blocks=ns/stride,err=0;
  size_t nd=3*(size_t)n,bytes=nd*sizeof(double);
  for(int p=0;p<m;++p) if(indices[p]<0||indices[p]>=n) return 1;
  for(size_t p=0;p<(size_t)blocks*m;++p) if(!isfinite(obs[p])) return 1;
  double *cp=calloc((size_t)(blocks+1)*nd,sizeof(double));
  if(!cp) return 2;
  *cost=INFINITY;
  err=swe_forward(initial,nx,ny,lx,ly,dt,ns,stride,cp,diag);
  if(err) {free(cp);return err;}
  *cost=0;
  for(int b=1;b<=blocks;++b) for(int p=0;p<m;++p) {
    double r=(cp[(size_t)b*nd+indices[p]]-obs[(size_t)(b-1)*m+p])/sigma;
    *cost+=0.5*r*r;
  }
  if(!gradient) {free(cp);return 0;}
  /* Checkpoint at each observation time; replay one block during reverse.
   * Memory: (blocks + 2*stride + 7) states, instead of 2*ns states. */
  double *buf=calloc((size_t)(2*stride+6)*nd,sizeof(double));
  if(!buf) {free(cp);return 2;}
  double *q=buf,*st=q+(size_t)(stride+1)*nd;
  Field tmp=field(st+(size_t)stride*nd,n);
  Field lam=field(tmp.h+nd,n),prev=field(lam.h+nd,n);
  Field work=field(prev.h+nd,n),work2=field(work.h+nd,n);
  Grid g={nx,ny,lx,ly,lx/nx,ly/ny};
  zero_field(&lam,n);
  for(int b=blocks;b>=1;--b) {
    for(int p=0;p<m;++p) {
      int k=indices[p];
      lam.h[k]+=(cp[(size_t)b*nd+k]-obs[(size_t)(b-1)*m+p])/(sigma*sigma);
    }
    memcpy(q,cp+(size_t)(b-1)*nd,bytes);
    for(int s=0;s<stride;++s) {
      Field u=field(q+(size_t)s*nd,n),v=field(q+(size_t)(s+1)*nd,n);
      Field stage=field(st+(size_t)s*nd,n);
      if((err=step_checked(&u,&stage,&tmp,&v,&g,dt,diag))) goto done;
    }
    for(int s=stride-1;s>=0;--s) {
      Field u=field(q+(size_t)s*nd,n),stage=field(st+(size_t)s*nd,n);
      rk2_adjoint_step(&u,&stage,&lam,&prev,&work,&work2,&g,dt);
      Field swap=lam;lam=prev;prev=swap;
    }
  }
  memcpy(gradient,lam.h,bytes);
done:
  free(buf);free(cp);return err;
}
