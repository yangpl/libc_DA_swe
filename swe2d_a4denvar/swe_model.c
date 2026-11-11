/* Checked forward-only API reused from swe2d_4dvar_tapenade.
 * No adjoint, checkpoint scheduler, or Tapenade runtime dependency. */
#include "swe_model.h"
#include "swe_primal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define G 9.81
#define HMIN 1.e-6
typedef struct { int nx, ny; double lx, ly, dx, dy; } Grid;
typedef struct { double *h, *hu, *hv; } Field;
static Field field(double *p,int n) { Field f={p,p+n,p+2*n}; return f; }
static void euler_update(const Field *u,Field *v,const Grid *g,double dt) {
  swe_euler(g->nx*g->ny,g->nx,g->ny,g->dx,g->dy,dt,u->h,v->h);
}
static int valid_args(int nx,int ny,double lx,double ly,double dt,int ns,int stride) {
  return nx>=2 && ny>=2 && nx<=4096 && ny<=4096 &&
    isfinite(lx) && lx>0 && isfinite(ly) && ly>0 && isfinite(dt) && dt>0 &&
    ns>0 && ns<=1000000 && stride>0 && ns%stride==0;
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
/* Bilinear cell-centre interpolation with constant boundary extension. */
static int receiver_xy(int indices[4],double weights[4],int nx,int ny,
                        double lx,double ly,double x,double y) {
  if(!isfinite(x)||!isfinite(y)||x<0||x>lx||y<0||y>ly) return 1;
  double gx=fmin(fmax((x/lx)*nx-.5,0.),nx-1.);
  double gy=fmin(fmax((y/ly)*ny-.5,0.),ny-1.);
  int i=(int)floor(gx),j=(int)floor(gy);
  if(i==nx-1) --i;
  if(j==ny-1) --j;
  double a=gx-i,b=gy-j;
  indices[0]=i+nx*j; indices[1]=i+1+nx*j;
  indices[2]=i+nx*(j+1); indices[3]=i+1+nx*(j+1);
  weights[0]=(1-a)*(1-b); weights[1]=a*(1-b);
  weights[2]=(1-a)*b; weights[3]=a*b;
  return 0;
}

int swe_sample_field(const double *values,int nx,int ny,double lx,double ly,
                     int m,const double *rx,const double *ry,double *samples) {
  if(!values||!rx||!ry||!samples||m<1||!valid_args(nx,ny,lx,ly,1.,1,1)) return 1;
  for(int p=0;p<m;++p) {
    int indices[4]; double weights[4];
    if(receiver_xy(indices,weights,nx,ny,lx,ly,rx[p],ry[p])) return 1;
    samples[p]=0;
    for(int q=0;q<4;++q)
      if(weights[q]!=0) samples[p]+=weights[q]*values[indices[q]];
    if(!isfinite(samples[p])) return 5;
  }
  return 0;
}

