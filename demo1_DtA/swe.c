#include "swe.h"

#define idx(i,j) ((i) + swe->nx*(j))

// ============================================================
void swe_step_forward(swe_t *swe, int it) {
  int i, j;
  float dtx = swe->dt / swe->dx;
  float dty = swe->dt / swe->dy;
  float uv, vu, Fu, Fv;
  float H_iph, H_imh, H_jph, H_jmh;

  // 保存轨迹 (仅当需要伴随的时候才真的需要存，这里为了简单每次都存)
  memcpy(&swe->z_store[it*swe->n], swe->z, swe->n * sizeof(float));
  memcpy(&swe->u_store[it*swe->n], swe->u, swe->n * sizeof(float));
  memcpy(&swe->v_store[it*swe->n], swe->v, swe->n * sizeof(float));

  // Update H = h + z
  for(i=0; i<swe->n; i++) swe->H[i] = swe->h[i] + swe->z[i];

  for(j=0; j<swe->ny; j++) {
    for(i=0; i<swe->nx; i++) {
      // 默认继承上一时刻的值，保证边界点也有定义
      swe->z_temp[idx(i,j)] = swe->z[idx(i,j)];
      swe->u_temp[idx(i,j)] = swe->u[idx(i,j)];
      swe->v_temp[idx(i,j)] = swe->v[idx(i,j)];
      Fu = swe->u[idx(i,j)];
      Fv = swe->v[idx(i,j)];

      // 计算内部点
      if(i-1>=0 && i+1<swe->nx && j-1>=0 && j+1<swe->ny) {
	// --- 连续方程 ---
	H_iph = 0.5*(swe->H[idx(i,j)] + swe->H[idx(i+1,j)]);
	H_imh = 0.5*(swe->H[idx(i,j)] + swe->H[idx(i-1,j)]);
	swe->z_temp[idx(i,j)] -= dtx*(H_iph*swe->u[idx(i,j)] - H_imh*swe->u[idx(i-1,j)]);

	H_jph = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j+1)]);
	H_jmh = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j-1)]);
	swe->z_temp[idx(i,j)] -= dty*(H_jph*swe->v[idx(i,j)] - H_jmh*swe->v[idx(i,j-1)]);

	// --- 动量平流项 ---
	// 修正后的 Fv (Central Difference)
	uv = 0.25*(swe->u[idx(i-1,j)] + swe->u[idx(i,j)] + swe->u[idx(i-1,j+1)] + swe->u[idx(i,j+1)]);
	Fv -= swe->dt*(uv*(swe->v[idx(i+1,j)]-swe->v[idx(i-1,j)])/(2*swe->dx)
		       + swe->v[idx(i,j)]*(swe->v[idx(i,j+1)]-swe->v[idx(i,j-1)])/(2*swe->dy));

	// Fu
	vu = 0.25*(swe->v[idx(i,j)] + swe->v[idx(i,j-1)] + swe->v[idx(i+1,j)] + swe->v[idx(i+1,j-1)]);
	Fu -= swe->dt*(swe->u[idx(i,j)]*(swe->u[idx(i+1,j)]-swe->u[idx(i-1,j)])/(2*swe->dx)
		       + vu*(swe->u[idx(i,j+1)]-swe->u[idx(i,j-1)])/(2*swe->dy));
      }

      // --- 动量方程线性项 ---
      if(i+1<swe->nx) {
	swe->u_temp[idx(i,j)] = Fu - G*dtx*(swe->z[idx(i+1,j)] - swe->z[idx(i,j)]) 
	  - swe->dt*swe->gamma*swe->u[idx(i,j)] + swe->f*swe->dt*swe->v[idx(i,j)];
      }
      if(j+1<swe->ny) {
	swe->v_temp[idx(i,j)] = Fv - G*dty*(swe->z[idx(i,j+1)] - swe->z[idx(i,j)]) 
	  - swe->dt*swe->gamma*swe->v[idx(i,j)] - swe->f*swe->dt*swe->u[idx(i,j)];
      }
    }
  }
  memcpy(swe->z, swe->z_temp, swe->n * sizeof(float));
  memcpy(swe->u, swe->u_temp, swe->n * sizeof(float));
  memcpy(swe->v, swe->v_temp, swe->n * sizeof(float));
}

void swe_step_adjoint(swe_t *swe, int it) {
  int i, j;
  float dtx = swe->dt / swe->dx;
  float dty = swe->dt / swe->dy;
  float dt = swe->dt;

  float *u_old = &swe->u_store[it * swe->n];
  float *v_old = &swe->v_store[it * swe->n];
  float *z_old = &swe->z_store[it * swe->n];

  // Input adjoints at t+1 (read-only snapshots)
  memcpy(swe->z_temp, swe->a_z, swe->n * sizeof(float));
  memcpy(swe->u_temp, swe->a_u, swe->n * sizeof(float));
  memcpy(swe->v_temp, swe->a_v, swe->n * sizeof(float));
  float *az_in = swe->z_temp;
  float *au_in = swe->u_temp;
  float *av_in = swe->v_temp;

  // Output adjoints at t (accumulators)
  memset(swe->a_z, 0, swe->n * sizeof(float));
  memset(swe->a_u, 0, swe->n * sizeof(float));
  memset(swe->a_v, 0, swe->n * sizeof(float));
  memset(swe->a_H, 0, swe->n * sizeof(float));

  for(i=0; i<swe->n; i++) swe->H[i] = swe->h[i] + z_old[i];

  // Identity inheritance from z_temp=z, u_temp=u, v_temp=v
  for(i=0; i<swe->n; i++) {
    swe->a_z[i] += az_in[i];
    swe->a_u[i] += au_in[i];
    swe->a_v[i] += av_in[i];
  }

  // Momentum linear terms
  for(j=0; j<swe->ny; j++) {
    for(i=0; i<swe->nx; i++) {
      int ij = idx(i,j);
      if(j+1 < swe->ny) {
        float a_Fv = av_in[ij];
        swe->a_z[idx(i,j+1)] += -G * dty * a_Fv;
        swe->a_z[ij]         +=  G * dty * a_Fv;
        swe->a_v[ij]         += -dt * swe->gamma * a_Fv;
        swe->a_u[ij]         += -swe->f * dt * a_Fv;
      }
      if(i+1 < swe->nx) {
        float a_Fu = au_in[ij];
        swe->a_z[idx(i+1,j)] += -G * dtx * a_Fu;
        swe->a_z[ij]         +=  G * dtx * a_Fu;
        swe->a_u[ij]         += -dt * swe->gamma * a_Fu;
        swe->a_v[ij]         +=  swe->f * dt * a_Fu;
      }
    }
  }

  // Momentum advection terms
  for(j=0; j<swe->ny; j++) {
    for(i=0; i<swe->nx; i++) {
      if(i-1>=0 && i+1<swe->nx && j-1>=0 && j+1<swe->ny) {
        int ij = idx(i,j);

        float common_v = -dt * av_in[ij];
        float coeff_diff_y = common_v * v_old[ij] / (2*swe->dy);
        swe->a_v[idx(i,j+1)] += coeff_diff_y;
        swe->a_v[idx(i,j-1)] -= coeff_diff_y;
        swe->a_v[ij]         += common_v * (v_old[idx(i,j+1)] - v_old[idx(i,j-1)]) / (2*swe->dy);

        float uv = 0.25*(u_old[idx(i-1,j)] + u_old[ij] + u_old[idx(i-1,j+1)] + u_old[idx(i,j+1)]);
        float coeff_diff_x = common_v * uv / (2*swe->dx);
        swe->a_v[idx(i+1,j)] += coeff_diff_x;
        swe->a_v[idx(i-1,j)] -= coeff_diff_x;

        float a_uv = common_v * (v_old[idx(i+1,j)] - v_old[idx(i-1,j)]) / (2*swe->dx);
        float a_u_part = 0.25 * a_uv;
        swe->a_u[idx(i-1,j)]   += a_u_part;
        swe->a_u[ij]           += a_u_part;
        swe->a_u[idx(i-1,j+1)] += a_u_part;
        swe->a_u[idx(i,j+1)]   += a_u_part;

        float common_u = -dt * au_in[ij];
        float coeff_diff_xu = common_u * u_old[ij] / (2*swe->dx);
        swe->a_u[idx(i+1,j)] += coeff_diff_xu;
        swe->a_u[idx(i-1,j)] -= coeff_diff_xu;
        swe->a_u[ij]         += common_u * (u_old[idx(i+1,j)] - u_old[idx(i-1,j)]) / (2*swe->dx);

        float vu = 0.25*(v_old[ij] + v_old[idx(i,j-1)] + v_old[idx(i+1,j)] + v_old[idx(i+1,j-1)]);
        float coeff_diff_yu = common_u * vu / (2*swe->dy);
        swe->a_u[idx(i,j+1)] += coeff_diff_yu;
        swe->a_u[idx(i,j-1)] -= coeff_diff_yu;

        float a_vu = common_u * (u_old[idx(i,j+1)] - u_old[idx(i,j-1)]) / (2*swe->dy);
        float a_v_part = 0.25 * a_vu;
        swe->a_v[ij]           += a_v_part;
        swe->a_v[idx(i,j-1)]   += a_v_part;
        swe->a_v[idx(i+1,j)]   += a_v_part;
        swe->a_v[idx(i+1,j-1)] += a_v_part;
      }
    }
  }

  // Continuity terms
  for(j=0; j<swe->ny; j++) {
    for(i=0; i<swe->nx; i++) {
      if(i-1>=0 && i+1<swe->nx && j-1>=0 && j+1<swe->ny) {
        int ij = idx(i,j);
        float common_z = az_in[ij];

        float H_iph = 0.5 * (swe->H[ij] + swe->H[idx(i+1,j)]);
        float H_imh = 0.5 * (swe->H[ij] + swe->H[idx(i-1,j)]);
        swe->a_u[ij]         += -dtx * H_iph * common_z;
        swe->a_u[idx(i-1,j)] +=  dtx * H_imh * common_z;

        float a_H_iph = -dtx * u_old[ij] * common_z;
        swe->a_H[ij]         += 0.5 * a_H_iph;
        swe->a_H[idx(i+1,j)] += 0.5 * a_H_iph;
        float a_H_imh = dtx * u_old[idx(i-1,j)] * common_z;
        swe->a_H[ij]         += 0.5 * a_H_imh;
        swe->a_H[idx(i-1,j)] += 0.5 * a_H_imh;

        float H_jph = 0.5 * (swe->H[ij] + swe->H[idx(i,j+1)]);
        float H_jmh = 0.5 * (swe->H[ij] + swe->H[idx(i,j-1)]);
        swe->a_v[ij]         += -dty * H_jph * common_z;
        swe->a_v[idx(i,j-1)] +=  dty * H_jmh * common_z;

        float a_H_jph = -dty * v_old[ij] * common_z;
        swe->a_H[ij]         += 0.5 * a_H_jph;
        swe->a_H[idx(i,j+1)] += 0.5 * a_H_jph;
        float a_H_jmh = dty * v_old[idx(i,j-1)] * common_z;
        swe->a_H[ij]         += 0.5 * a_H_jmh;
        swe->a_H[idx(i,j-1)] += 0.5 * a_H_jmh;
      }
    }
  }

  for(i=0; i<swe->n; i++) swe->a_z[i] += swe->a_H[i];
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
