//https://gemini.google.com/share/dee09f4c01cb
#include "swe.h"

// 辅助函数保持不变
double* alloc1double(int n) {
    double *p = (double*)malloc(n * sizeof(double));
    if (!p) { fprintf(stderr, "Memory allocation error\n"); exit(1); }
    memset(p, 0, n * sizeof(double));
    return p;
}
void free1double(double* p) { if(p) free(p); }
// 向量点积函数
double dot_product(double *v1, double *v2, int n) {
    double sum = 0.0;
    for(int i=0; i<n; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}
// Forward Step 保持不变
void swe_step_forward(swe_t *swe, double *z_new, double *u_new, double *v_new) {
    // ... (保持您原有的前向代码不变) ...
    // 为了节省篇幅，此处省略，逻辑与之前提供的一致
    int i, j;
    double dtx = swe->dt / swe->dx;
    double dty = swe->dt / swe->dy;
    double uv, vu, Fu, Fv;
    double H_iph, H_imh, H_jph, H_jmh;

    for(i=0; i<swe->n; i++) swe->H[i] = swe->h[i] + swe->z[i];

    for(j=0; j<swe->ny; j++) {
        for(i=0; i<swe->nx; i++) {
            z_new[idx(i,j)] = swe->z[idx(i,j)];
            u_new[idx(i,j)] = swe->u[idx(i,j)];
            v_new[idx(i,j)] = swe->v[idx(i,j)];
            
            Fu = swe->u[idx(i,j)];
            Fv = swe->v[idx(i,j)];

            if(i-1>=0 && i+1<swe->nx && j-1>=0 && j+1<swe->ny) {
                H_iph = 0.5*(swe->H[idx(i,j)] + swe->H[idx(i+1,j)]);
                H_imh = 0.5*(swe->H[idx(i,j)] + swe->H[idx(i-1,j)]);
                z_new[idx(i,j)] -= dtx*(H_iph*swe->u[idx(i,j)] - H_imh*swe->u[idx(i-1,j)]);

                H_jph = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j+1)]);
                H_jmh = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j-1)]);
                z_new[idx(i,j)] -= dty*(H_jph*swe->v[idx(i,j)] - H_jmh*swe->v[idx(i,j-1)]);

                uv = 0.25*(swe->u[idx(i-1,j)] + swe->u[idx(i,j)] + swe->u[idx(i-1,j+1)] + swe->u[idx(i,j+1)]);
                Fv -= swe->dt*(uv*(swe->v[idx(i+1,j)]-swe->v[idx(i-1,j)])/(2*swe->dx)
                             + swe->v[idx(i,j)]*(swe->v[idx(i,j+1)]-swe->v[idx(i,j-1)])/(2*swe->dy));

                vu = 0.25*(swe->v[idx(i,j)] + swe->v[idx(i,j-1)] + swe->v[idx(i+1,j)] + swe->v[idx(i+1,j-1)]);
                Fu -= swe->dt*(swe->u[idx(i,j)]*(swe->u[idx(i+1,j)]-swe->u[idx(i-1,j)])/(2*swe->dx)
                             + vu*(swe->u[idx(i,j+1)]-swe->u[idx(i,j-1)])/(2*swe->dy));
            }

            if(i+1<swe->nx) {
                u_new[idx(i,j)] = Fu - G*dtx*(swe->z[idx(i+1,j)] - swe->z[idx(i,j)]) 
                                  - swe->dt*swe->gamma*swe->u[idx(i,j)] + swe->f*swe->dt*swe->v[idx(i,j)];
            }
            if(j+1<swe->ny) {
                v_new[idx(i,j)] = Fv - G*dty*(swe->z[idx(i,j+1)] - swe->z[idx(i,j)]) 
                                  - swe->dt*swe->gamma*swe->v[idx(i,j)] - swe->f*swe->dt*swe->u[idx(i,j)];
            }
        }
    }
    
}

// ============================================================
// 伴随模型主循环 (包含正则化项)
// ============================================================
// 新增参数:
// - alpha_b: 正则化系数 (论文公式20中的 alpha_b) 
// - x_b_u, x_b_v, x_b_z: 背景状态 (通常设为0或先验估计) 
void swe_run_adjoint(swe_t *swe, double *diff_z_all_steps, double *background_z) {
    int i, j, it;
    double dt = swe->dt;
    // 假设背景场 (Background State) 为 0 (静止状态)，如论文 6.2.2.1 所述 
    // 如果需要非零背景场，可以作为参数传入
    double alpha_b = 0.0;
    double *background_u = alloc1double(swe->n);
    double *background_v = alloc1double(swe->n);
    
    double *dz_adj = alloc1double(swe->n);
    double *du_adj = alloc1double(swe->n);
    double *dv_adj = alloc1double(swe->n);

    // 初始化伴随变量
    memset(background_u, 0, swe->n * sizeof(double));
    memset(background_v, 0, swe->n * sizeof(double));
    memset(swe->a_z, 0, swe->n * sizeof(double));
    memset(swe->a_u, 0, swe->n * sizeof(double));
    memset(swe->a_v, 0, swe->n * sizeof(double));

    // 1. 逆时间循环 (计算观测项梯度 nabla J_obs)
    for(it = swe->nt - 1; it >= 0; it--) {
        
        // --- 观测强迫 (Obs Forcing) ---
        double *current_diff = &diff_z_all_steps[it * swe->n];
        for(i=0; i<swe->n; i++) {
            swe->a_z[i] += dt * current_diff[i]; 
        }

        // 获取背景场 (Trajectory)
        double *u = &swe->u_store[it * swe->n];
        double *v = &swe->v_store[it * swe->n];
        double *z = &swe->z_store[it * swe->n];
        for(i=0; i<swe->n; i++) swe->H[i] = swe->h[i] + z[i];

        memset(dz_adj, 0, swe->n * sizeof(double));
        memset(du_adj, 0, swe->n * sizeof(double));
        memset(dv_adj, 0, swe->n * sizeof(double));

        // --- 伴随更新量 (RHS) ---
        for(j = 1; j < swe->ny - 1; j++) {
            for(i = 1; i < swe->nx - 1; i++) {
                
                // z* Update
                double div_u_star = (swe->a_u[idx(i,j)] - swe->a_u[idx(i-1,j)]) / swe->dx;
                double div_v_star = (swe->a_v[idx(i,j)] - swe->a_v[idx(i,j-1)]) / swe->dy;
                
                double adv_x_term = (u[idx(i,j)]   * (swe->a_z[idx(i+1,j)] - swe->a_z[idx(i,j)]) + 
                                     u[idx(i-1,j)] * (swe->a_z[idx(i,j)]   - swe->a_z[idx(i-1,j)])) / (2.0 * swe->dx);
                double adv_y_term = (v[idx(i,j)]   * (swe->a_z[idx(i,j+1)] - swe->a_z[idx(i,j)]) + 
                                     v[idx(i,j-1)] * (swe->a_z[idx(i,j)]   - swe->a_z[idx(i,j-1)])) / (2.0 * swe->dy);

                dz_adj[idx(i,j)] = dt * ( G * (div_u_star + div_v_star) + adv_x_term + adv_y_term );

                // u* Update
                if (i < swe->nx - 1) {
                    double H_at_u = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i+1,j)]);
                    double grad_z_star = (swe->a_z[idx(i+1,j)] - swe->a_z[idx(i,j)]) / swe->dx;
                    double u_val = u[idx(i,j)]; 
                    double v_val = 0.25 * (v[idx(i,j)] + v[idx(i,j-1)] + v[idx(i+1,j)] + v[idx(i+1,j-1)]);
                    double dx_ustar = (swe->a_u[idx(i+1,j)] - swe->a_u[idx(i-1,j)]) / (2.0 * swe->dx);
                    double dy_ustar = (swe->a_u[idx(i,j+1)] - swe->a_u[idx(i,j-1)]) / (2.0 * swe->dy);
                    double v_star_avg = 0.25 * (swe->a_v[idx(i,j)] + swe->a_v[idx(i,j-1)] + swe->a_v[idx(i+1,j)] + swe->a_v[idx(i+1,j-1)]);
                    double dx_v = (0.5*(v[idx(i+1,j)]+v[idx(i+1,j-1)]) - 0.5*(v[idx(i,j)]+v[idx(i,j-1)])) / swe->dx;
                    
                    du_adj[idx(i,j)] = dt * (H_at_u * grad_z_star + u_val * dx_ustar + v_val * dy_ustar - 
                        v_star_avg * dx_v - swe->gamma * swe->a_u[idx(i,j)] - swe->f * v_star_avg);
                }

                // v* Update
                if (j < swe->ny - 1) {
                    double H_at_v = 0.5 * (swe->H[idx(i,j)] + swe->H[idx(i,j+1)]);
                    double grad_z_star_y = (swe->a_z[idx(i,j+1)] - swe->a_z[idx(i,j)]) / swe->dy;
                    double v_val_v = v[idx(i,j)];
                    double u_val_v = 0.25 * (u[idx(i,j)] + u[idx(i-1,j)] + u[idx(i,j+1)] + u[idx(i-1,j+1)]);
                    double dx_vstar = (swe->a_v[idx(i+1,j)] - swe->a_v[idx(i-1,j)]) / (2.0 * swe->dx);
                    double dy_vstar = (swe->a_v[idx(i,j+1)] - swe->a_v[idx(i,j-1)]) / (2.0 * swe->dy);
                    double u_star_avg = 0.25 * (swe->a_u[idx(i,j)] + swe->a_u[idx(i-1,j)] + swe->a_u[idx(i,j+1)] + swe->a_u[idx(i-1,j+1)]);
                    double dy_u = (0.5*(u[idx(i,j+1)]+u[idx(i-1,j+1)]) - 0.5*(u[idx(i,j)]+u[idx(i-1,j)])) / swe->dy;
                    
                    dv_adj[idx(i,j)] = dt * (H_at_v * grad_z_star_y + u_val_v * dx_vstar + v_val_v * dy_vstar -
                        u_star_avg * dy_u - swe->gamma * swe->a_v[idx(i,j)] + swe->f * u_star_avg);
                }
            }
        }

        for(i=0; i<swe->n; i++) {
            swe->a_z[i] += dz_adj[i];
            swe->a_u[i] += du_adj[i];
            swe->a_v[i] += dv_adj[i];
        }

        // Boundary conditions
        for(j=0; j<swe->ny; j++) {
            swe->a_u[idx(0,j)] = 0; swe->a_u[idx(swe->nx-1,j)] = 0;
            swe->a_v[idx(0,j)] = 0; swe->a_v[idx(swe->nx-1,j)] = 0;
            swe->a_z[idx(0,j)] = 0; swe->a_z[idx(swe->nx-1,j)] = 0;
        }
        for(i=0; i<swe->nx; i++) {
            swe->a_u[idx(i,0)] = 0; swe->a_u[idx(i,swe->ny-1)] = 0;
            swe->a_v[idx(i,0)] = 0; swe->a_v[idx(i,swe->ny-1)] = 0;
            swe->a_z[idx(i,0)] = 0; swe->a_z[idx(i,swe->ny-1)] = 0;
        }

    } // End Time Loop

    // ============================================================
    // 2. 加入正则化项/背景项梯度 (Gradient of Regularization)
    // 根据论文公式 (14) 和 (20) [cite: 419, 582]
    // 梯度 nabla J = nabla J_obs + alpha_b * (x_0 - x_b)
    // 此时 swe->a_* 存储的是 nabla J_obs (在 t=0 时刻)
    // 我们需要将 alpha_b * (x_0 - x_b) 加到这些梯度上。
    // 注意：x_0 存储在 u_store, v_store, z_store 的第 0 个时间步
    // ============================================================
    
    double *u_0 = &swe->u_store[0]; // Initial Condition u
    double *v_0 = &swe->v_store[0]; // Initial Condition v
    double *z_0 = &swe->z_store[0]; // Initial Condition z

    for(i=0; i<swe->n; i++) {
        // 如果是边界，通常不进行正则化更新（保持为0），或者根据 B 矩阵处理
        // 这里假设简单 Tikhonov 正则化
        
        // u component
        swe->a_u[i] += alpha_b * (u_0[i] - background_u[i]);
        
        // v component
        swe->a_v[i] += alpha_b * (v_0[i] - background_v[i]);
        
        // z component
        swe->a_z[i] += alpha_b * (z_0[i] - background_z[i]);
    }
    

    free1double(dz_adj);
    free1double(du_adj);
    free1double(dv_adj);
}
