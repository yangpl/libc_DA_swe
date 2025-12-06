#ifndef __swe__
#define __swe__

typedef struct {
  int verb;//verbosity

  int niter;//number of CG iterations
  double tol;//tolerance for convergence

  double f;//Coriolis friction
  double gamma;//constant coefficient
  int nt;//number of time steps
  int nx;
  int ny;
  int n; //n=nx*ny
  double dt;
  double dx;
  double dy;
  double *h;
  double *z;
  double *u;
  double *v;
  double *H;//H=z+h
  double *zz;
  double *uu;
  double *vv;
  
  double *b;//right hand side for pentadiagonal system
} swe_t;//data structure for shallow water equation (SWE)

#endif
