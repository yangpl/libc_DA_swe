/* Direct tests of conservation and the full-state transpose, including edges.
 * Include the implementation to test internal Euler and RK2 operators. */
#include "reference/manual_swe.c"
#include <stdio.h>
int main(void) {
  const int nx=9,ny=7,n=63,nd=189;
  Grid grid={nx,ny,1.3,.8,1.3/nx,.8/ny};
  double storage[12*189]={0};
  Field u=field(storage,n),v=field(storage+nd,n),d=field(storage+2*nd,n);
  Field l=field(storage+3*nd,n),a=field(storage+4*nd,n);
  Field up=field(storage+5*nd,n),um=field(storage+6*nd,n);
  Field vp=field(storage+7*nd,n),vm=field(storage+8*nd,n);
  double dt=.001,eps=1e-6;
  for(int k=0;k<nd;++k) {
    u.h[k]=(k<n ? 1.0 : 0.02)+.01*sin(k*.73);
    d.h[k]=cos(k*.71);l.h[k]=sin(k*.31);
    up.h[k]=u.h[k]+eps*d.h[k];um.h[k]=u.h[k]-eps*d.h[k];
  }
  euler_update(&u,&v,&grid,dt);
  double change[3]={0},boundary[3]={0};
  for(int c=0;c<3;++c) for(int k=0;k<n;++k) change[c]+=(v.h[c*n+k]-u.h[c*n+k])*grid.dx*grid.dy;
  for(int j=0;j<ny;++j) {
    int lidx=j*nx,ridx=j*nx+nx-1;
    State ql={u.h[lidx],u.hu[lidx],u.hv[lidx]},qr={u.h[ridx],u.hu[ridx],u.hv[ridx]};
    State fl=flux_x(ql),fr=flux_x(qr);
    boundary[0]-=dt*grid.dy*(fr.h-fl.h);boundary[1]-=dt*grid.dy*(fr.hu-fl.hu);boundary[2]-=dt*grid.dy*(fr.hv-fl.hv);
  }
  for(int i=0;i<nx;++i) {
    int b=i,t=(ny-1)*nx+i;
    State qb={u.h[b],u.hu[b],u.hv[b]},qt={u.h[t],u.hu[t],u.hv[t]};
    State fb=flux_y(qb),ft=flux_y(qt);
    boundary[0]-=dt*grid.dx*(ft.h-fb.h);boundary[1]-=dt*grid.dx*(ft.hu-fb.hu);boundary[2]-=dt*grid.dx*(ft.hv-fb.hv);
  }
  for(int c=0;c<3;++c) if(fabs(change[c]-boundary[c])>1e-14) return 1;
  euler_update(&up,&vp,&grid,dt);euler_update(&um,&vm,&grid,dt);
  euler_adjoint(&u,&l,&a,&grid,dt);
  double lhs=0,rhs=0;
  for(int k=0;k<nd;++k) {lhs+=(vp.h[k]-vm.h[k])/(2*eps)*l.h[k];rhs+=a.h[k]*d.h[k];}
  double rel=fabs(lhs-rhs)/fmax(1.,fabs(rhs));
  printf("Euler full-state transpose relative error %.3e; open-boundary conservation passed\n",rel);
  if(rel>1e-7) return 1;
  double observations[12],gradient[189],diagnostics[2],cost,jp,jm;
  int indices[4]={0,8,54,62};
  for(int p=0;p<12;++p) observations[p]=1.+.002*cos(p);
  if(swe_evaluate(u.h,nx,ny,1.3,.8,dt,12,4,4,indices,observations,.02,&cost,gradient,diagnostics)) return 2;
  if(swe_evaluate(up.h,nx,ny,1.3,.8,dt,12,4,4,indices,observations,.02,&jp,NULL,diagnostics)) return 2;
  if(swe_evaluate(um.h,nx,ny,1.3,.8,dt,12,4,4,indices,observations,.02,&jm,NULL,diagnostics)) return 2;
  rhs=0;
  for(int k=0;k<nd;++k) rhs+=gradient[k]*d.h[k];
  rel=fabs((jp-jm)/(2*eps)-rhs)/fmax(1.,fabs(rhs));
  printf("Checkpointed observation-adjoint relative error %.3e\n",rel);
  return rel>1e-7;
}
