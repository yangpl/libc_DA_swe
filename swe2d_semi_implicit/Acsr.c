#include "cstd.h"
#include "sparse.h"
#include "swe.h"

#define G 9.81

csr_t *A;

//allocate CSR matrix A
void Acsr_alloc(swe_t *swe)
{
  int i, j, k;  
  //count the number of nonzeros in sparse banded matrix A
  k = 0;
  for(j=0; j<swe->ny; j++){
    for(i=0; i<swe->nx; i++){
      //the diagonal element A(i,j)
      k++;
      
      //the off-diagonal element
      if(j-1>=0){//element A(i,j-1)
	k++;	
      }
      if(i-1>=0){//element A(i-1,j)
	k++;	
      }
      if(i+1<swe->nx){//element A(i+1,j)
	k++;
      }
      if(j+1<swe->ny){//element A(i,j+1)
	k++;
      }      
    }
  }
  A = (csr_t*)malloc(sizeof(csr_t));
  A->nnz = k;//number of non-zeros
  A->nrow = swe->nx*swe->ny;//the same as swe->n
  A->ncol = swe->nx*swe->ny;//the same as swe->n
  
  A->row_ptr = alloc1int(A->nrow+1);
  A->col_ind = alloc1int(A->nnz);
  A->val = alloc1double(A->nnz);
}

void Acsr_free(swe_t *swe)
{
  free1int(A->row_ptr);
  free1int(A->col_ind);
  free1double(A->val);
  free(A);
}

//build matrix in CSR format for matrix operator A
void Acsr_init(swe_t *swe, double *b)
{
  int i, j, k, row_ind;
  double dtx = swe->dt/swe->dx;
  double dty = swe->dt/swe->dy;
  double g_dtx2 = G*dtx*dtx;
  double g_dty2 = G*dty*dty;
  double coeff, diag;
  double H_imh, H_iph, H_jmh, H_jph;
  
#define idx(i,j) (i + swe->nx*(j))
  
  for(j=0; j<swe->ny; j++){
    for(i=0; i<swe->nx; i++){
      swe->H[idx(i,j)] = swe->h[idx(i,j)] + swe->z[idx(i,j)];//H=z+h at (ij)
      
      swe->Fu[idx(i,j)] = swe->u[idx(i,j)];
      swe->Fv[idx(i,j)] = swe->v[idx(i,j)];
      if(i-1>=0 && i+1<swe->nx && j-1>=0 && j+1<swe->ny){//advection term (u\partial_x + v\partial_y), switch from center to upwind scheme later !!!
	double vu = 0.25*(swe->v[idx(i,j)] + swe->v[idx(i,j-1)] + swe->v[idx(i+1,j)] + swe->v[idx(i+1,j-1)]);//v at u[i+0.5,j]
	swe->Fu[idx(i,j)] -= swe->dt*swe->u[idx(i,j)]*(swe->u[idx(i+1,j)]-swe->u[idx(i-1,j)])/(2*swe->dx);//upwind
	swe->Fu[idx(i,j)] -= swe->dt*vu*(swe->u[idx(i,j+1)]-swe->u[idx(i,j-1)])/(2*swe->dy);

	double uv = 0.25*(swe->u[idx(i-1,j)] + swe->u[idx(i,j)] + swe->u[idx(i-1,j+1)] + swe->u[idx(i,j+1)]);//u at v[i,j+0.5]
	swe->Fv[idx(i,j)] -= swe->dt*uv*(swe->v[idx(i,j)]-swe->v[idx(i-1,j)])/(2*swe->dx);
	swe->Fv[idx(i,j)] -= swe->dt*swe->v[idx(i,j)]*(swe->v[idx(i,j+1)]-swe->v[idx(i,j-1)])/(2*swe->dy);
      }
    }
  }

  k = 0;
  for(j=0; j<swe->ny; j++){
    for(i=0; i<swe->nx; i++){
      diag = 1.0;
      H_imh = 0;
      H_iph = 0;
      H_jmh = 0;
      H_jph = 0;
      
      //the off-diagonal element
      if(i-1>=0){//element A(i-1,j)
        H_imh = 0.5*(swe->H[idx(i,j)] + swe->H[idx(i-1,j)])/(1.0 + swe->gamma * swe->dt);//H(i-0.5,j)/(1+gamma*dt)
        coeff = g_dtx2 * H_imh;
	A->val[k] = -coeff;
	A->col_ind[k] = idx(i-1,j);
	k++;

        diag += coeff;
      }
      if(i+1<swe->nx){//element A(i+1,j)
        H_iph = 0.5*(swe->H[idx(i,j)] + swe->H[idx(i+1,j)])/(1.0 + swe->gamma * swe->dt);//H(i+0.5,j)/(1+gamma*dt)
        coeff = g_dtx2 * H_iph;
	A->val[k] = -coeff;
	A->col_ind[k] = idx(i+1,j);
	k++;

	diag += coeff;
      }
      if(j-1>=0){//element A(i,j-1)
        H_jmh = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j-1)])/(1.0 + swe->gamma * swe->dt);//H(i,j-0.5)
        coeff = g_dty2 * H_jmh;
	A->val[k] = -coeff;
	A->col_ind[k] = idx(i,j-1);
	k++;	

        diag += coeff;
      }
      if(j+1<swe->ny){//element A(i,j+1)
        H_jph = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j+1)])/(1.0 + swe->gamma * swe->dt);//H(i,j+0.5)
        coeff = g_dty2 * H_jph;
	A->val[k] = -coeff;
	A->col_ind[k] = idx(i,j+1);
	k++;

	diag += coeff;
      }
      
      //the diagonal element A(i,j) and b
      A->val[k] = diag;//TO BE modified
      A->col_ind[k] = idx(i,j);
      k++;
      
      row_ind = idx(i,j);//row index
      A->row_ptr[row_ind+1] = k;

      //right hand side at row idx(i,j)
      b[idx(i,j)] = swe->z[idx(i,j)];
      if (i-1>=0 && i+1< swe->nx && j-1>=0 && j+1<swe->ny){
	b[idx(i,j)] -= dtx*( H_iph*(swe->Fu[idx(i,j)] - swe->dt*swe->f*swe->Fv[idx(i,j)])
			     -H_imh*(swe->Fu[idx(i-1,j)] - swe->dt*swe->f*swe->Fv[idx(i-1,j)]) );
	b[idx(i,j)] -= dty*( H_jph*(swe->Fv[idx(i,j)] + swe->dt*swe->f*swe->Fu[idx(i,j)])
			     -H_jmh*(swe->Fv[idx(i,j-1)] + swe->dt*swe->f*swe->Fu[idx(i,j-1)]) );
      }

    }
  }

#undef idx
}


//compute y=Ax by applying operator A in CSR format
void Acsr_apply(int n, double *x, double *y)
{
  int i, j, k;

  for(i=0; i<A->nrow; i++){
    y[i] = 0;
    for(k=A->row_ptr[i]; k<A->row_ptr[i+1]; k++){
      j = A->col_ind[k];
      y[i] += A->val[k]*x[j];
    }//end k
  }//end i

}
