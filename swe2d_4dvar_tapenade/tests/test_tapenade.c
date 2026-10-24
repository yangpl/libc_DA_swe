/* Independent manual oracle plus signed directional differences for generated AD. */
#include "../swe2d_4dvar.h"
#define swe_forward reference_forward
#define swe_evaluate reference_evaluate
#include "reference/manual_swe.c"
#undef swe_forward
#undef swe_evaluate
#include <stdio.h>
int main(void) {
  enum {nx=9,ny=7,n=63,nd=189};
  double x[nd], saved[nd],g[nd],ref[nd],obs[33*4],diag[2],j,jref;
  int idx[]={0,8,54,62}, budgets[]={1,2,4,8,90}, lengths[]={1,7,12,33};
  double worst=0;
  for(int k=0;k<nd;++k) saved[k]=x[k]=(k<n?1.:.02)+.01*sin(k*.73);
  for(int k=0;k<33*4;++k) obs[k]=1.+.02*cos(k*.37);
  for(int t=0;t<4;++t) for(int b=0;b<5;++b) {
    int ns=lengths[t], stride=(ns%3==0?3:1);
    if(reference_evaluate(x,nx,ny,1.3,.8,.001,ns,stride,4,idx,obs,.02,&jref,ref,diag)) return 1;
    if(swe_evaluate_binomial(x,nx,ny,1.3,.8,.001,ns,stride,4,idx,obs,.02,budgets[b],&j,g,diag)) return 2;
    if(j!=jref) return 3;
    double err=0,norm=0;
    for(int k=0;k<nd;++k) {err+=(g[k]-ref[k])*(g[k]-ref[k]);norm+=ref[k]*ref[k];}
    double rel=sqrt(err/norm);
    worst=fmax(worst,rel);
    if(rel>2e-12 || memcmp(x,saved,sizeof(x))) return 4;
    if(b==0 || b==3) printf("nt=%d budget=%d relative=%.3e\n", ns,budgets[b],rel);
    double xp[nd],xm[nd],jp,jm,adj=0,eps=1e-6;
    for(int k=0;k<nd;++k) {double d=cos(k*.71);xp[k]=x[k]+eps*d;xm[k]=x[k]-eps*d;adj+=g[k]*d;}
    if(swe_evaluate(xp,nx,ny,1.3,.8,.001,ns,stride,4,idx,obs,.02,&jp,NULL,diag) ||
       swe_evaluate(xm,nx,ny,1.3,.8,.001,ns,stride,4,idx,obs,.02,&jm,NULL,diag)) return 5;
    if(fabs((jp-jm)/(2*eps)-adj)/fmax(1.,fabs(adj))>2e-7) return 6;
  }
  for(int budget=-1;budget<=91;budget+=92)
    if(swe_evaluate_binomial(x,nx,ny,1.3,.8,.001,12,3,4,idx,obs,.02,budget,&j,g,diag)!=1) return 7;
  /* Cross several 1 MiB tape blocks, including checkpoint repeat/restore. */
  {
    const int big_n=3600,big_nd=10800;
    double *big=calloc(3*(size_t)big_nd,sizeof(double));
    if(!big) return 8;
    double *bg=big+big_nd,*br=big+2*big_nd;
    for(int k=0;k<big_nd;++k) big[k]=(k<big_n?1.:.02)+.01*sin(k*.73);
    int sensors[]={0,59,3540,3599};
    if(reference_evaluate(big,60,60,1.,1.,.001,12,3,4,sensors,obs,.02,&jref,br,diag) ||
       swe_evaluate_binomial(big,60,60,1.,1.,.001,12,3,4,sensors,obs,.02,8,&j,bg,diag)) return 9;
    for(int k=0;k<big_nd;++k) if(fabs(bg[k]-br[k])>1e-11*fmax(1.,fabs(br[k]))) return 10;
    if(j!=jref) return 11;
    if(swe_evaluate(big,60,60,1.,1.,.001,12,3,4,sensors,obs,.02,&j,bg,diag)) return 12;
    for(int k=0;k<big_nd;++k) if(fabs(bg[k]-br[k])>1e-11*fmax(1.,fabs(br[k]))) return 13;
    free(big);
    puts("60x60 multi-block tape and legacy C API passed");
  }
  printf("Tapenade/manual worst relative gradient error %.3e; finite differences passed\n",worst);
  return 0;
}
