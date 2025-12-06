#include "cstd.h"
#include "swe.h"
#include "solver.h"

#define G 9.81

#define idx(i,j) (i + swe->nx*(j))

void Acsr_alloc(swe_t *swe);
void Acsr_init(swe_t *swe, double *b);
void Acsr_free(swe_t *swe);
void Acsr_apply(int n, double *x, double *y);

void solve_cg(int n, double *x, double *b, op_t Aop, int niter, double tol, int verb);

int main(int argc, char *argv[])
{
  swe_t *swe;
  int i, j, it;
  
  initargs(argc, argv);  
  swe = (swe_t *)malloc(sizeof(swe_t));

  if(!getparint("verb", &swe->verb)) swe->verb = 0;//verbosity
  if(!getparint("nt", &swe->nt)) swe->nt = 500;//number of time steps
  if(!getparint("nx", &swe->nx)) swe->nx = 101;
  if(!getparint("ny", &swe->ny)) swe->ny = 101;
  if(!getpardouble("dx", &swe->dx)) swe->dx = 10.;
  if(!getpardouble("dy", &swe->dy)) swe->dy = 10.;
  if(!getpardouble("dt", &swe->dt)) swe->dt = 0.1;
  if(!getparint("niter", &swe->niter)) swe->niter = 1000;//maximum number of iterations
  if(!getpardouble("tol", &swe->tol)) swe->tol = 1e-12;
  if(!getpardouble("gamma", &swe->gamma)) swe->gamma = 0;//constant coefficient
  if(!getpardouble("f", &swe->f)) swe->f = 0;//Coriolis friction
  
  swe->n = swe->nx*swe->ny;     
  swe->h = alloc1double(swe->n);
  swe->z = alloc1double(swe->n);
  swe->u = alloc1double(swe->n);
  swe->v = alloc1double(swe->n); 
  swe->Fu = alloc1double(swe->n);
  swe->Fv = alloc1double(swe->n); 
  swe->H = alloc1double(swe->n);
  swe->b = alloc1double(swe->n);
  
  double xc = (swe->nx - 1) * swe->dx / 2.0;
  double yc = (swe->ny - 1) * swe->dy / 2.0;
  double sigma_x = swe->nx * swe->dx / 10.0;
  double sigma_y = swe->ny * swe->dy / 10.0;
  for(j = 0; j < swe->ny; j++) {
    for(i = 0; i < swe->nx; i++) {
      double dx = i*swe->dx - xc;
      double dy = j*swe->dy - yc;
      swe->z[idx(i,j)] = 0.5 * exp(-(dx*dx)/(2*sigma_x*sigma_x) - (dy*dy)/(2*sigma_y*sigma_y));
      swe->h[idx(i,j)] = 10.0;
    }
  }
  memset(swe->u, 0, swe->n*sizeof(double));
  memset(swe->v, 0, swe->n*sizeof(double)); 
    
  FILE *fp = fopen("zxy.bin", "w");
  Acsr_alloc(swe);//allocate A (in CSR format), b (and uu and vv)
  for(it=0; it < swe->nt; it++) {
    printf("it=%d\n", it);

    //initialize CSR matrix A, right hand side b,
    Acsr_init(swe, swe->b);

    //solve (symmetric positive definite) pentadiagonal system using conjugate gradient method
    solve_cg(swe->n, swe->z, swe->b, Acsr_apply, swe->niter, swe->tol, swe->verb);
    fwrite(swe->z, swe->n*sizeof(double), 1, fp);
        
    //update velocity fields u and v in the interior
    for(j=0; j<swe->ny-1; j++) {
      for(i=0; i<swe->nx-1; i++) {
	swe->u[idx(i,j)] = swe->Fu[idx(i,j)] - G*swe->dt*(swe->z[idx(i+1,j)] - swe->z[idx(i,j)])/swe->dx
	  - swe->dt*swe->gamma*swe->u[idx(i,j)] - swe->f*swe->dt*swe->Fv[idx(i,j)];
	swe->v[idx(i,j)] = swe->Fv[idx(i,j)] - G*swe->dt*(swe->z[idx(i,j+1)] - swe->z[idx(i,j)])/swe->dy
	  - swe->dt*swe->gamma*swe->v[idx(i,j)] + swe->f*swe->dt*swe->Fu[idx(i,j)];
	  
	  //if(i<swe->nb || i>swe->nx-nb) swe->u[idx(i,j)] *= swe->bndr[]
      }
    }
    //handling the boundaries: set normal velocity=0
  }
  fclose(fp);
  Acsr_free(swe);

  free1double(swe->b);
  free1double(swe->H);
  free1double(swe->h);
  free1double(swe->z);
  free1double(swe->u);
  free1double(swe->v);
  free1double(swe->Fu);
  free1double(swe->Fv);
    
  return 0;
}
