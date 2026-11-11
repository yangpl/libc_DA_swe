/* Native forward-model checks, also run under ASan/UBSan. */
#include "swe_model.h"
#include "swe_primal.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

int main(void) {
  enum { NX=7, NY=5, N=NX*NY, NS=3*N };
  double q[NS], out[5*NS], diag[2], euler[NS];
  for(int k=0;k<N;++k) {q[k]=1.2;q[N+k]=.08;q[2*N+k]=-.04;}
  assert(swe_forward(q,NX,NY,1.,.8,.001,12,3,out,diag)==0);
  for(int t=0;t<5;++t) for(int k=0;k<NS;++k) assert(fabs(out[t*NS+k]-q[k])<3e-15);
  assert(diag[0]>1. && diag[1]<.9);
  for(int k=0;k<N;++k) {
    q[k]=1.1+.01*sin((double)k);q[N+k]=.02*cos((double)k);q[2*N+k]=.01*sin(2.*k);
  }
  /* Sum of each conservative Euler increment equals net physical boundary flux. */
  swe_euler(N,NX,NY,1./NX,.8/NY,.001,q,euler);
  double expected[3]={0}, actual[3]={0};
  for(int axis=0;axis<2;++axis) {
    int along=axis ? NX : NY;
    for(int b=0;b<along;++b) {
      int left=axis ? b : NX*b;
      int right=axis ? b+NX*(NY-1) : NX*b+NX-1;
      double ql[3],qr[3],fl[3],fr[3];
      for(int c=0;c<3;++c) {ql[c]=q[c*N+left];qr[c]=q[c*N+right];}
      swe_flux(axis,ql,ql,fl);swe_flux(axis,qr,qr,fr);
      for(int c=0;c<3;++c) expected[c]+=.001*(fl[c]-fr[c])/(axis ? .8/NY : 1./NX);
    }
  }
  for(int c=0;c<3;++c) {
    for(int k=0;k<N;++k) actual[c]+=euler[c*N+k]-q[c*N+k];
    assert(fabs(actual[c]-expected[c])<2e-14);
  }
  assert(swe_forward(q,NX,NY,1.,.8,.001,12,3,out,diag)==0);
  q[0]=0.;assert(swe_forward(q,NX,NY,1.,.8,.001,12,3,out,diag)==3);
  q[0]=NAN;assert(swe_forward(q,NX,NY,1.,.8,.001,12,3,out,diag)==3);
  q[0]=1.;assert(swe_forward(q,NX,NY,1.,.8,1.,12,3,out,diag)==4);
  assert(swe_forward(q,1,NY,1.,.8,.001,12,3,out,diag)==1);
  assert(swe_forward(q,NX,NY,1.,.8,.001,12,0,out,diag)==1);
  assert(swe_forward(NULL,NX,NY,1.,.8,.001,12,3,out,diag)==1);
  for(int j=0;j<NY;++j) for(int i=0;i<NX;++i) {
    double x=(i+.5)/NX,y=(j+.5)*.8/NY;
    q[i+NX*j]=1.+2*x-3*y+4*x*y;
  }
  double rx[4]={.31,.61,0.,1.},ry[4]={.27,.53,0.,.8},samples[4];
  assert(swe_sample_field(q,NX,NY,1.,.8,4,rx,ry,samples)==0);
  for(int k=0;k<2;++k) assert(fabs(samples[k]-(1.+2*rx[k]-3*ry[k]+4*rx[k]*ry[k]))<1e-14);
  assert(samples[2]==q[0] && samples[3]==q[N-1]);
  rx[0]=-1.;assert(swe_sample_field(q,NX,NY,1.,.8,4,rx,ry,samples)==1);
  puts("Uniform flow, all-component boundary-flux conservation, stage rejection, and bilinear receivers passed");
  return 0;
}
