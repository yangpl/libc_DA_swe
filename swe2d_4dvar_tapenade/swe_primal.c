/* Tapenade input: the actual wet-state forward model and scalar DA misfit.
 * Allocation and validity checks belong to the public C wrapper. All array
 * extents are explicit; see fix_tapenade_c.py for Tapenade C output repairs.
 * No derivative routines are supplied to Tapenade. */
#include <math.h>
#include "swe_primal.h"

static double swe_abs(double x) {
  if (x > 0.0) return x;
  if (x < 0.0) return -x;
  return 0.0;
}

void swe_flux(int axis, const double left[3], const double right[3], double flux[3]) {
  int c;
  double hl=left[0], hr=right[0];
  double ul=left[1]/hl, vl=left[2]/hl;
  double ur=right[1]/hr, vr=right[2]/hr;
  double fl[3],fr[3],al,ar,speed;
  if (axis == 0) {
    fl[0]=left[1]; fl[1]=left[1]*ul+0.5*9.81*hl*hl; fl[2]=left[1]*vl;
    fr[0]=right[1]; fr[1]=right[1]*ur+0.5*9.81*hr*hr; fr[2]=right[1]*vr;
    al=swe_abs(ul)+sqrt(9.81*hl); ar=swe_abs(ur)+sqrt(9.81*hr);
  } else {
    fl[0]=left[2]; fl[1]=left[2]*ul; fl[2]=left[2]*vl+0.5*9.81*hl*hl;
    fr[0]=right[2]; fr[1]=right[2]*ur; fr[2]=right[2]*vr+0.5*9.81*hr*hr;
    al=swe_abs(vl)+sqrt(9.81*hl); ar=swe_abs(vr)+sqrt(9.81*hr);
  }
  speed=(al>=ar ? al : ar);
  for (c=0;c<3;++c)
    flux[c]=0.5*(fl[c]+fr[c])-0.5*speed*(right[c]-left[c]);
}

void swe_euler(int n,int nx,int ny,double dx,double dy,double dt,
               const double input[3*n],double output[3*n]) {
  int k,axis,across,along,b,a,lo,hi,l=0,r=0,c;
  double scale=0.0,ql[3]={0},qr[3]={0},flux[3];
  for (k=0;k<3*n;++k) output[k]=input[k];
  for (axis=0;axis<2;++axis) {
    across=(axis ? ny : nx); along=(axis ? nx : ny);
    scale=dt/(axis ? dy : dx);
    for (b=0;b<along;++b) for (a=0;a<=across;++a) {
      lo=(a>0 ? a-1 : 0); hi=(a<across ? a : across-1);
      l=(axis ? b+nx*lo : lo+nx*b); r=(axis ? b+nx*hi : hi+nx*b);
      for (c=0;c<3;++c) {ql[c]=input[c*n+l];qr[c]=input[c*n+r];}
      swe_flux(axis,ql,qr,flux);
      for (c=0;c<3;++c) {
        if (a>0) output[c*n+l]-=scale*flux[c];
        if (a<across) output[c*n+r]+=scale*flux[c];
      }
    }
  }
}

void swe_rk2(int n,int nx,int ny,double dx,double dy,double dt,
             double state[3*n],double stage[3*n],double tmp[3*n]) {
  int k;
  swe_euler(n,nx,ny,dx,dy,dt,state,stage);
  swe_euler(n,nx,ny,dx,dy,dt,stage,tmp);
  for (k=0;k<3*n;++k) state[k]=0.5*(state[k]+tmp[k]);
}

/* Keep the residual calculation in a routine: it is replayed locally by AD. */
void swe_observation(int n,int m,int step,int stride,const int indices[m],
                      const double observations[],double sigma,
                      const double state[3*n],double *cost) {
  int p,cycle=(step+1)/stride-1;
  double residual;
  for (p=0;p<m;++p) {
    residual=(state[indices[p]]-observations[cycle*m+p])/sigma;
    *cost+=0.5*residual*residual;
  }
}

void swe_misfit(int n,int nx,int ny,double dx,double dy,double dt,
                int nt,int stride,int m,int nsnap,const int indices[m],
                const double observations[(nt/stride)*m],double sigma,
                const double initial[3*n],double state[3*n],
                double stage[3*n],double tmp[3*n],double *cost) {
  int k,it;
  (void)nsnap; /* Used by the Tapenade directive below. */
  for (k=0;k<3*n;++k) state[k]=initial[k];
  *cost=0.0;
  /* $AD BINOMIAL-CKP nt nsnap 0 */
  for (it=0;it<nt;++it) {
    swe_rk2(n,nx,ny,dx,dy,dt,state,stage,tmp);
    if ((it+1)%stride==0)
      swe_observation(n,m,it,stride,indices,observations,sigma,state,cost);
  }
}
