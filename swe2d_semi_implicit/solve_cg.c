/* - conjugate gradient method
 *------------------------------------------------------------------------
 *
 * Copyright (c) 2020-2022 Harbin Institute of Technology. All rights reserved.
 * Author: Pengliang Yang 
 * Email: ypl.2100@gmail.com
 * Homepage: https://yangpl.wordpress.com
 *-----------------------------------------------------------------------*/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "solver.h"

//compute dot product s=<x,y>
double dotprod(int n, double *a, double *b)
{
  int i;
  double s;
  
  for(i=0, s=0; i<n; i++) s += a[i]*b[i];
  return s;  
}

//linear solver using conjugate gradient method
void solve_cg(int n, double *x, double *b, op_t Aop, int niter, double tol, int verb)
{
  int i, k;
  double rsold, rsnew, rs0, pAp, alp, beta;

  double *r = malloc(n*sizeof(double));
  double *p = malloc(n*sizeof(double));
  double *Ap = malloc(n*sizeof(double));

  Aop(n, x, Ap);//Ap=Ax0
  for(i=0; i<n; i++) {
    r[i] = b[i]-Ap[i];//r=b-Ax
    p[i] = r[i];
  }
  rsold = dotprod(n, r, r);
  rs0 = rsold;

  for(k=0; k<niter; k++){
    if(verb) printf("CG, k=%d rs=%e\n", k, rsold);

    Aop(n, p, Ap);//Ap=A*p
    pAp = dotprod(n, p, Ap);
    alp = rsold/pAp;

    for(i=0; i<n; i++){
      x[i] += alp*p[i];
      r[i] -= alp*Ap[i];
    }
    rsnew = dotprod(n, r, r);
    if(sqrt(rsnew)<tol*sqrt(rs0)) {
      if(verb) printf("converged at k=%d\n", k);
      break;
    }
    
    beta = rsnew/rsold;
    for(i=0; i<n; i++) p[i] = r[i] + beta*p[i];
    rsold = rsnew;
  }

  free(r);
  free(p);
  free(Ap);
}

//linear solver using conjugate gradient method
void solve_pcg(int n, double *x, double *b, op_t Aop, op_t invMop, int niter, double tol, int verb)
{
  int i, k;
  double rzold, rznew, rs0, pAp, alp, beta, rs;

  double *r = malloc(n*sizeof(double));
  double *p = malloc(n*sizeof(double));
  double *Ap = malloc(n*sizeof(double));
  double *z = malloc(n*sizeof(double));

  Aop(n, x, Ap);//Ap=Ax0
  for(i=0; i<n; i++) r[i] = b[i]-Ap[i];//r=b-Ax
  invMop(n, r, z);
  memcpy(p, z, n*sizeof(double));
  rzold = dotprod(n, r, z);
  rs = dotprod(n, r, r);
  rs = sqrt(rs);
  rs0 = rs;

  for(k=0; k<niter; k++){
    if(verb) printf("PCG, k=%d rs=%e\n", k, rs);

    Aop(n, p, Ap);//Ap=A*p
    pAp = dotprod(n, p, Ap);
    alp = rzold/pAp;

    for(i=0; i<n; i++){
      x[i] += alp*p[i];
      r[i] -= alp*Ap[i];
    }
    invMop(n, r, z);
    rznew = dotprod(n, r, z);
    rs = dotprod(n, r, r);
    rs = sqrt(rs);
    if(rs<tol*rs0) {
      if(verb) printf("converged at k=%d\n", k);
      break;
    }
    
    beta = rznew/rzold;
    for(i=0; i<n; i++) p[i] = z[i] + beta*p[i];
    rzold = rznew;
  }

  free(r);
  free(p);
  free(Ap);
  free(z);
}
