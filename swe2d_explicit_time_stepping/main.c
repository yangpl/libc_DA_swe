#include "cstd.h"
#include "swe.h"

#define G 9.81
#define idx(i,j) (i + swe->nx*(j))

int main(int argc, char *argv[])
{
  swe_t *swe;
  int i, j, it;
  double uv, vu, dtx, dty, Fu, Fv, flux_x, flux_y;
  double H_iph, H_imh, H_jph, H_jmh;
  double *ptr;
  
  initargs(argc, argv);  
  swe = (swe_t *)malloc(sizeof(swe_t));

  if(!getparint("verb", &swe->verb)) swe->verb = 0;
  if(!getparint("nt", &swe->nt)) swe->nt = 500;
  if(!getparint("nx", &swe->nx)) swe->nx = 101;
  if(!getparint("ny", &swe->ny)) swe->ny = 101;
  if(!getpardouble("dx", &swe->dx)) swe->dx = 10.;
  if(!getpardouble("dy", &swe->dy)) swe->dy = 10.;
  if(!getpardouble("dt", &swe->dt)) swe->dt = 0.1;
  if(!getparint("niter", &swe->niter)) swe->niter = 1000;
  if(!getpardouble("tol", &swe->tol)) swe->tol = 1e-12;
  if(!getpardouble("gamma", &swe->gamma)) swe->gamma = 0;
  if(!getpardouble("f", &swe->f)) swe->f = 0;
  
  swe->n = swe->nx*swe->ny;     
  swe->h = alloc1double(swe->n);
  swe->z = alloc1double(swe->n);
  swe->u = alloc1double(swe->n);
  swe->v = alloc1double(swe->n); 
  swe->H = alloc1double(swe->n);
  swe->zz = alloc1double(swe->n);
  swe->uu = alloc1double(swe->n);
  swe->vv = alloc1double(swe->n);

  double xc = (swe->nx-1) * swe->dx / 2.0;
  double yc = (swe->ny-1) * swe->dy / 2.0;
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

  dtx = swe->dt/swe->dx;
  dty = swe->dt/swe->dy;
  FILE *fp = fopen("zxy.bin", "w");
  
  for(it=0; it < swe->nt; it++) {
    if(it%100 == 0) printf("it=%d\n", it); 

    // 1. compute total water depth H(x,y,t)=h(x,y) + z(x,y,t)
    for(j=0; j<swe->ny; j++){
      for(i=0; i<swe->nx; i++){
        swe->H[idx(i,j)] = swe->h[idx(i,j)] + swe->z[idx(i,j)];
      }
    }
    // 2. compute (u,v,h)^{it+1} from (u,v,h)^{it}
    for(j=0; j<swe->ny; j++){
      for(i=0; i<swe->nx; i++){
        Fu = swe->u[idx(i,j)];
        Fv = swe->v[idx(i,j)];
	if(i-1>=0 && i+1<swe->nx && j-1>=0 && j+1<swe->ny){
	  vu = 0.25*(swe->v[idx(i,j)] + swe->v[idx(i,j-1)] + swe->v[idx(i+1,j)] + swe->v[idx(i+1,j-1)]);
	  Fu -= swe->dt*(swe->u[idx(i,j)]*(swe->u[idx(i+1,j)]-swe->u[idx(i-1,j)])/(2*swe->dx)
			 + vu*(swe->u[idx(i,j+1)]-swe->u[idx(i,j-1)])/(2*swe->dy));

	  uv = 0.25*(swe->u[idx(i-1,j)] + swe->u[idx(i,j)] + swe->u[idx(i-1,j+1)] + swe->u[idx(i,j+1)]);
	  Fv -= swe->dt*(uv*(swe->v[idx(i,j)]-swe->v[idx(i-1,j)])/(2*swe->dx)
			 + swe->v[idx(i,j)]*(swe->v[idx(i,j+1)]-swe->v[idx(i,j-1)])/(2*swe->dy));
	}
	//update u and v
	if(i+1<swe->nx){
	  swe->uu[idx(i,j)] = Fu - G*dtx*(swe->z[idx(i+1,j)] - swe->z[idx(i,j)]) - swe->dt*swe->gamma*swe->u[idx(i,j)] - swe->f*swe->dt*Fv;
	}else{
	  swe->uu[idx(i,j)] = swe->u[idx(i,j)];//preserve values on the boundary
	}
	if(j+1<swe->ny){
	  swe->vv[idx(i,j)] = Fv - G*dty*(swe->z[idx(i,j+1)] - swe->z[idx(i,j)]) - swe->dt*swe->gamma*swe->v[idx(i,j)] + swe->f*swe->dt*Fu;
	}else{
	  swe->vv[idx(i,j)] = swe->v[idx(i,j)];//preserve values on the boundary
	}

	//update z based on continuity equation
	if(i-1>=0 && i+1<swe->nx && j-1>=0 && j+1<swe->ny){
	  H_iph = 0.5*(swe->H[idx(i,j)] + swe->H[idx(i+1,j)]);
	  H_imh = 0.5*(swe->H[idx(i,j)] + swe->H[idx(i-1,j)]);
	  flux_x = dtx*(H_iph*swe->u[idx(i,j)] - H_imh*swe->u[idx(i-1,j)]);

	  H_jph = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j+1)]);
	  H_jmh = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j-1)]);
	  flux_y = dty*(H_jph*swe->v[idx(i,j)] - H_jmh*swe->v[idx(i,j-1)]);
	  swe->zz[idx(i,j)] = swe->z[idx(i,j)] - flux_x - flux_y;
	}else
	  swe->zz[idx(i,j)] = swe->z[idx(i,j)];//preseve values on the boundary
      }
    }
    // 3. update (z,u,v) by swaping pointers for (z,u,v)^{it} and (z,u,v)^{it+1} 
    ptr = swe->u; swe->u = swe->uu; swe->uu = ptr;
    ptr = swe->v; swe->v = swe->vv; swe->vv = ptr;
    ptr = swe->z; swe->z = swe->zz; swe->zz = ptr;

    fwrite(swe->z, swe->n*sizeof(double), 1, fp);
  }
  fclose(fp);
  
  free1double(swe->H);
  free1double(swe->h);
  free1double(swe->z);
  free1double(swe->u);
  free1double(swe->v);
  free1double(swe->zz);
  free1double(swe->uu);
  free1double(swe->vv);
    
  return 0;
}
