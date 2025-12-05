#include "cstd.h"
#include "swe.h"

#define idx(i,j) ((i) + swe->nx*(j))

void swe_step_forward(swe_t *swe, int it);
void swe_step_adjoint(swe_t *swe, int it); 

//initial state uses 2D Gaussian function
void init_gaussian(swe_t *swe, float shift_x_grid, float shift_y_grid, float delta) {
  int i, j;
  float xc = (swe->nx-1) * swe->dx / 2.0 + shift_x_grid * swe->dx;
  float yc = (swe->ny-1) * swe->dy / 2.0 + shift_y_grid * swe->dy;
  float sigma = swe->nx * swe->dx / delta; 
  for(j = 0; j < swe->ny; j++) {
    for(i = 0; i < swe->nx; i++) {
      float dx = i*swe->dx - xc;
      float dy = j*swe->dy - yc;
      swe->z[idx(i,j)] = 1.0 * exp(-(dx*dx + dy*dy) / (2*sigma*sigma));
      swe->h[idx(i,j)] = 10.0; 
      swe->u[idx(i,j)] = 0; 
      swe->v[idx(i,j)] = 0;
    }
  }
}

//compute cost function and its gradient g according to state vector x
float compute_cost(swe_t *swe, float *x, float *g)
{
  int i, j, k;
  int it;
  float cost = 0.0;
  
  //initialize forward state variables to 0
  memcpy(swe->z, x, swe->n * sizeof(float));
  memset(swe->u, 0, swe->n * sizeof(float));
  memset(swe->v, 0, swe->n * sizeof(float));    
  for(it=0; it < swe->nt; it++) {//forward time integration for forward state
    k = 0;
    for(j=0; j<swe->ny; j++){
      for(i=0; i<swe->nx; i++){
	if (i % swe->stride_x == 0 && j % swe->stride_y == 0) {
	  swe->z_cal[k + it*swe->np] = swe->z[idx(i,j)];
	  swe->z_res[k + it*swe->np] = swe->z_cal[k + it*swe->np] - swe->z_obs[k + it*swe->np];
	  cost += 0.5*swe->z_res[k + it*swe->np]*swe->z_res[k + it*swe->np];
	  k++;
	}
      }
    }
    swe_step_forward(swe, it);
  }//end for it
  
  //initialize adjoint state variables to 0
  memset(swe->a_z, 0, swe->n * sizeof(float));
  memset(swe->a_u, 0, swe->n * sizeof(float));
  memset(swe->a_v, 0, swe->n * sizeof(float));
  memset(swe->a_H, 0, swe->n * sizeof(float));
  for(it = swe->nt - 1; it >= 0; it--) {//backward time integration for adjoint state
    k = 0;
    for(j=0; j<swe->ny; j++){
      for(i=0; i<swe->nx; i++){
	if (i % swe->stride_x == 0 && j % swe->stride_y == 0) {
	  swe->a_z[idx(i,j)] += swe->z_res[k + it*swe->np];
	  k++;
	}
      }
    }
    swe_step_adjoint(swe, it);
  }//end for it
  memcpy(g, swe->a_z, swe->n*sizeof(float));//gradient g=a_z
  
  return cost;
}

int main(int argc, char *argv[])
{
  int i, j, k, it;
  int initopt, iter, niter;
  swe_t *swe;
  FILE *fp;
  
  initargs(argc, argv);

  swe = (swe_t *)malloc(sizeof(swe_t));
  if(!getparint("initopt", &initopt)) initopt = 0; // 0: 平坦场, 1: 偏移的真实场    
  if(!getparint("niter", &niter)) niter = 100; //maximum number of iterations
  if(!getparint("nt", &swe->nt)) swe->nt = 100;
  if(!getparint("nx", &swe->nx)) swe->nx = 101;
  if(!getparint("ny", &swe->ny)) swe->ny = 101;
  if(!getparfloat("dt", &swe->dt)) swe->dt = 0.1;
  if(!getparfloat("dx", &swe->dx)) swe->dx = 10.;
  if(!getparfloat("dy", &swe->dy)) swe->dy = 10.;
  if(!getparfloat("gamma", &swe->gamma)) swe->gamma = 0.0;
  if(!getparfloat("f", &swe->f)) swe->f = 0.0;
  if(!getparint("stride_x", &swe->stride_x)) swe->stride_x = 5;
  if(!getparint("stride_y", &swe->stride_y)) swe->stride_y = 5;
  swe->n = swe->nx * swe->ny;
  swe->np = (swe->nx/swe->stride_x) * (swe->ny/swe->stride_y);
    
  swe->h = malloc(swe->n*sizeof(float));
  swe->H = malloc(swe->n*sizeof(float));
  swe->z = malloc(swe->n*sizeof(float));
  swe->u = malloc(swe->n*sizeof(float));
  swe->v = malloc(swe->n*sizeof(float));
  swe->a_z = malloc(swe->n*sizeof(float));
  swe->a_u = malloc(swe->n*sizeof(float));
  swe->a_v = malloc(swe->n*sizeof(float));
  swe->a_H = malloc(swe->n*sizeof(float));
  swe->z_store = malloc(swe->nt*swe->n*sizeof(float));
  swe->u_store = malloc(swe->nt*swe->n*sizeof(float));
  swe->v_store = malloc(swe->nt*swe->n*sizeof(float));
  swe->z_obs = malloc(swe->nt*swe->np*sizeof(float));
  swe->z_cal = malloc(swe->nt*swe->np*sizeof(float));
  swe->z_res = malloc(swe->nt*swe->np*sizeof(float));
  swe->z_temp = malloc(swe->n*sizeof(float));
  swe->u_temp = malloc(swe->n*sizeof(float));
  swe->v_temp = malloc(swe->n*sizeof(float));

  // ============================================================
  printf("--- Phase 1: generate true observations by running SWE nt steps\n");
  init_gaussian(swe, 0.0, 0.0, 15);
  fp = fopen("z_true_init.bin", "wb");
  fwrite(swe->z, swe->n*sizeof(float), 1, fp);
  fclose(fp);
  for(it=0; it < swe->nt; it++) {
    k = 0;
    for(j=0; j<swe->ny; j++){
      for(i=0; i<swe->nx; i++){
	if (i % swe->stride_x == 0 && j % swe->stride_y == 0) {
	  swe->z_obs[k + it*swe->np] = swe->z[idx(i,j)];//record observed data
	  k++;
	}
      }
    }
    swe_step_forward(swe, it);
  }

  // ============================================================
  float *x = malloc(swe->n*sizeof(float)); //state vector at current iteration
  float *g = malloc(swe->n*sizeof(float));  // g_k
  float *d = malloc(swe->n*sizeof(float)); // d_k (descent direction)
  float *gold = malloc(swe->n*sizeof(float));  // g_{k-1}
  float *xx = malloc(swe->n*sizeof(float)); //test state vector in line search
  float *gg = malloc(swe->n*sizeof(float)); //gradient vector at test state vector 
  printf("--- Phase 2: Initialization ---\n");
  if(initopt == 0) {//initialize z with flat field
    for(j = 0; j < swe->ny; j++) {
      for(i = 0; i < swe->nx; i++) {
	swe->z[idx(i,j)] = 0.0;
	swe->h[idx(i,j)] = 10.0; 
      }
    }
    memcpy(x, swe->z, swe->n * sizeof(float));
  } else {//initialize field z with a different Gaussian distribution
    init_gaussian(swe, 3.0, 3.0, 10);
    memcpy(x, swe->z, swe->n * sizeof(float));
  }    
  fp = fopen("z_guess_init.bin", "wb");
  fwrite(swe->z, swe->n*sizeof(float), 1, fp);
  fclose(fp);

  // ============================================================
  printf("--- Phase 3: Polak-Ribiere NLCG Optimization ---\n");
  float fx, fxx, fx0;
  float alpha, beta, num, den;
  fx = compute_cost(swe, x, g);
  fx0 = fx;
  for(i=0; i<swe->n; i++) d[i] = -g[i];//descent direction d=-g
  memcpy(gold, g, swe->n * sizeof(float));//backup gradient vector

  FILE *fp_cost = fopen("cost_history.txt", "w");
  alpha = 1.;
  for(iter = 0; iter < niter; iter++) {
    printf("Iter %2d: Cost = %.6e ", iter, fx);
    fprintf(fp_cost, "%d %.10e\n", iter, fx); // 修复：添加分号
        
    if(fx<1e-4*fx0) { 
      printf("\nConverged!\n"); 
      break; 
    }

    //A: line search, 1st iteration alpha=1, take alpha from previous iteration after
    const float c1 = 1e-4; // Armijo (Sufficient Decrease) parameter
    const float c2 = 0.9;  // Curvature (Sufficient Slope) parameter (typically 0.1 to 0.9)
    float slope = 0;
    for(i=0; i<swe->n; i++) slope += g[i]*d[i]; // Original slope: g[x] dot d

    int ls_iter = 0;
    while(ls_iter < 20) { // line search <= 20 times
      // 1. test state vector: xx = x + alpha * d
      for(i=0; i<swe->n; i++) xx[i] = x[i] + alpha * d[i];
        
      // Compute cost f(xx) and gradient gg = grad f(xx)
      // NOTE: The compute_cost function must now store the new gradient in gg (or whatever 'g' corresponds to)
      // Assuming compute_cost(swe, xx, gg) computes f(xx) and fills gg with grad f(xx).
      // For the purpose of line search, we need the new gradient gg.
      // We will temporarily use the 'g' array for the new gradient gg, as 'g' will be overwritten 
      // in the main loop anyway. We'll use a new array `gg` for clarity here.
      fxx = compute_cost(swe, xx, gg); // fxx is f(x + alpha*d), gg is grad f(x + alpha*d)

      // 2. check Armijo condition (Sufficient Decrease)
      if(fxx > fx + c1 * alpha * slope) {
	// Armijo failed: Step is too large, function value didn't decrease enough.
	alpha *= 0.5;
	ls_iter++;
	continue; // Continue to next iteration with smaller alpha
      }

      // 3. check curvature condition (Sufficient Slope Increase)
      float new_slope = 0;
      for(i=0; i<swe->n; i++) new_slope += gg[i]*d[i]; // New slope: grad f(xx) dot d

      if(new_slope >= c2 * slope) {
	// Curvature satisfied: The slope at the new point is sufficiently less negative (or more positive).
	// Both Wolfe conditions satisfied, break the loop.
	break; 
      } else {
	// Curvature failed: Step is too short, the slope is still too negative.
	// Increase the step length (e.g., double it, or use interpolation logic)
	// A simple doubling might be agoldressive; interpolation is better, but for simplicity:
	alpha *= 2.0; 
	ls_iter++;
      }
    }
    printf("| Alpha = %.4e | New Cost = %.6e | ls=%d\n", alpha, fxx, ls_iter);

    //B: update state vector x and gradient vector g after line search
    memcpy(x, xx, swe->n * sizeof(float));
    memcpy(g, gg, swe->n * sizeof(float)); 
    fx = fxx;

    //C: compute beta in Polak-Ribiere NLCG algorithm
    num = 0;
    den = 0;
    for(i=0; i<swe->n; i++){
      num += g[i]*(g[i] - gold[i]);
      den += gold[i]*gold[i];
    }
    beta = fmax(num / den, 0);    
    for(i=0; i<swe->n; i++) d[i] = -g[i] + beta * d[i];//NLCG
    memcpy(gold, g, swe->n * sizeof(float));//g^{k-1}<---g^k

    //output latest x at each iteration
    fp = fopen("z_final_analysis.bin", "wb");
    fwrite(x, swe->n*sizeof(float), 1, fp);
    fclose(fp);
  }
  fclose(fp_cost);

  free(swe->h);
  free(swe->H);
  free(swe->z); 
  free(swe->u);
  free(swe->v);
  free(swe->a_z); 
  free(swe->a_u);
  free(swe->a_v);
  free(swe->a_H);
  free(swe->z_store);
  free(swe->u_store);
  free(swe->v_store);
  free(swe->z_obs);
  free(swe->z_cal);
  free(swe->z_res); 
  free(swe->z_temp);
  free(swe->u_temp);
  free(swe->v_temp);
  free(swe);

  free(x);
  free(g);
  free(gold);
  free(d);
  free(xx);
  free(gg);

  return 0;
}
