#include "swe.h"

// 辅助函数：保存二进制文件
void save_to_file(const char *filename, double *data, int n) {
    FILE *fp = fopen(filename, "wb");
    if(!fp) return;
    fwrite(data, sizeof(double), n, fp);
    fclose(fp);
}

// 初始化高斯波
void init_gaussian(swe_t *swe, double shift_x_grid, double shift_y_grid, double delta) {
    int i, j;
    double xc = (swe->nx-1) * swe->dx / 2.0 + shift_x_grid * swe->dx;
    double yc = (swe->ny-1) * swe->dy / 2.0 + shift_y_grid * swe->dy;
    double sigma = swe->nx * swe->dx / delta; 
    for(j = 0; j < swe->ny; j++) {
        for(i = 0; i < swe->nx; i++) {
            double dx = i*swe->dx - xc;
            double dy = j*swe->dy - yc;
            swe->z[idx(i,j)] = 1.0 * exp(-(dx*dx + dy*dy) / (2*sigma*sigma));
            swe->h[idx(i,j)] = 10.0; 
            swe->u[idx(i,j)] = 0; swe->v[idx(i,j)] = 0;
        }
    }
}

// --- 封装 Cost 计算函数 (用于线搜索) ---
double compute_cost(swe_t *swe, double *z_current, double *diff_z, int obs_stride) {
    // 1. 设置初始状态
    memcpy(swe->z, z_current, swe->n * sizeof(double));
    memset(swe->u, 0, swe->n * sizeof(double));
    memset(swe->v, 0, swe->n * sizeof(double));
    
    // 临时变量
    double *z_n = alloc1double(swe->n);
    double *u_n = alloc1double(swe->n);
    double *v_n = alloc1double(swe->n);
    double cost = 0.0;

    // 2. 积分循环
    for(int it=0; it < swe->nt; it++) {
        // 保存轨迹 (仅当需要伴随的时候才真的需要存，这里为了简单每次都存)
        memcpy(&swe->z_store[it*swe->n], swe->z, swe->n * sizeof(double));
        memcpy(&swe->u_store[it*swe->n], swe->u, swe->n * sizeof(double));
        memcpy(&swe->v_store[it*swe->n], swe->v, swe->n * sizeof(double));

        double *obs_ptr = &swe->z_obs[it*swe->n];
        double *diff_ptr = &diff_z[it*swe->n];

        // 稀疏观测逻辑
        for(int j=0; j<swe->ny; j++) {
            for(int i=0; i<swe->nx; i++) {
                int k = idx(i, j);
                if (i % obs_stride == 0 && j % obs_stride == 0) {
                    diff_ptr[k] = swe->z[k] - obs_ptr[k];
                    cost += 0.5 * diff_ptr[k] * diff_ptr[k];
                } else {
                    diff_ptr[k] = 0.0;
                }
            }
        }
        swe_step_forward(swe, z_n, u_n, v_n);
        memcpy(swe->z, z_n, swe->n * sizeof(double));
        memcpy(swe->u, u_n, swe->n * sizeof(double));
        memcpy(swe->v, v_n, swe->n * sizeof(double));
    }
    free1double(z_n); free1double(u_n); free1double(v_n);
    return cost;
}

int main(int argc, char *argv[]) {
    swe_t *swe = (swe_t *)malloc(sizeof(swe_t));
    
    // 配置参数
    swe->nx = 101; swe->ny = 101; swe->nt = 100;
    swe->dx = 10.0; swe->dy = 10.0; swe->dt = 0.1;
    swe->gamma = 0.0; swe->f = 0.0; 
    swe->n = swe->nx * swe->ny;
    
    // 内存分配
    swe->h = alloc1double(swe->n); swe->H = alloc1double(swe->n);
    swe->z = alloc1double(swe->n); swe->u = alloc1double(swe->n); swe->v = alloc1double(swe->n);
    swe->a_z = alloc1double(swe->n); swe->a_u = alloc1double(swe->n);
    swe->a_v = alloc1double(swe->n); swe->a_H = alloc1double(swe->n);
    swe->z_store = alloc1double(swe->nt * swe->n);
    swe->u_store = alloc1double(swe->nt * swe->n);
    swe->v_store = alloc1double(swe->nt * swe->n);
    swe->z_obs   = alloc1double(swe->nt * swe->n);
    double *diff_z = alloc1double(swe->nt * swe->n);

    // ============================================================
    // Step 1: 生成真实观测 (True Run)
    // ============================================================
    double *z_temp = alloc1double(swe->n);
    double *u_temp = alloc1double(swe->n);
    double *v_temp = alloc1double(swe->n);
    double *z_true_init = alloc1double(swe->n); // 保存真实初始场

    printf("--- Phase 1: Generating Truth ---\n");
    init_gaussian(swe, 0.0, 0.0, 15);
    memcpy(z_true_init, swe->z, swe->n * sizeof(double)); // 保存真实初始场
    save_to_file("z_true_init.bin", swe->z, swe->n);

    for(int it=0; it < swe->nt; it++) {
        memcpy(&swe->z_obs[it * swe->n], swe->z, swe->n * sizeof(double));
        swe_step_forward(swe, z_temp, u_temp, v_temp);
        memcpy(swe->z, z_temp, swe->n * sizeof(double));
        memcpy(swe->u, u_temp, swe->n * sizeof(double));
        memcpy(swe->v, v_temp, swe->n * sizeof(double));
    }

    // ============================================================
    // Step 2: 初始化 (Guess)
    // ============================================================
    // Step 2.1: 初始化观测场
    double *z_est = alloc1double(swe->n); // 当前最优估计
    
    // 选择初始化方式：平坦场或随机扰动
    int initialization_method = 2; // 0: 平坦场, 1: 随机扰动 ，2: 偏移的真实场    
    if (initialization_method == 0) {
        printf("--- Phase 2: Initialization with Flat Field ---\n");
        for(int j = 0; j < swe->ny; j++) {
            for(int i = 0; i < swe->nx; i++) {
                swe->z[idx(i,j)] = 0.0;
                swe->h[idx(i,j)] = 10.0; 
            }
        }
        memcpy(z_est, swe->z, swe->n * sizeof(double));
    } else if (initialization_method == 1) {
        printf("--- Phase 2: Initialization with Random Noise ---\n");
        // 从真实场复制并添加随机扰动
        memcpy(z_est, z_true_init, swe->n * sizeof(double)); 
        double noise_amplitude = 0.05; 
        
        // 对每个点基于其自身值的百分比添加扰动
        for(int k = 0; k < swe->n; k++) {
            double base_value = fabs(z_true_init[k]);
            // 避免除零，设置一个最小基准值
            if (base_value < 1e-6) base_value = 1e-6;
            
            double point_noise_amplitude = 0.1 * base_value;
            double random_val = (2.0 * rand() / (double)RAND_MAX) - 1.0;
            z_est[k] += point_noise_amplitude * random_val;
        }
    } else {
        init_gaussian(swe, 3.0, 3.0, 10);
        memcpy(z_est, swe->z, swe->n * sizeof(double));
    }
    
    save_to_file("z_guess_init.bin", z_est, swe->n);
    
    // step2.2: 初始化背景场
    double *background_z = alloc1double(swe->n);
    init_gaussian(swe, 0.0, 0.0, 12);
    memcpy(background_z, swe->z, swe->n * sizeof(double));
    // ============================================================
    // Step 3: CG 优化循环 (Conjugate Gradient)
    // ============================================================
    printf("--- Phase 3: Polak-Ribiere CG Optimization ---\n");

    // CG 需要的额外变量
    double *grad_old = alloc1double(swe->n);  // g_{k-1}
    double *direction = alloc1double(swe->n); // d_k (搜索方向)
    double *z_new_trial = alloc1double(swe->n); // 用于线搜索的试探点

    // 初始梯度
    int obs_stride = 3; // 稀疏观测间隔
    double current_cost = compute_cost(swe, z_est, diff_z, obs_stride);
    swe_run_adjoint(swe, diff_z, background_z); // 算出 g_0 -> swe->a_z

    // 初始方向 d_0 = -g_0
    for(int i=0; i<swe->n; i++) direction[i] = -swe->a_z[i];
    
    // 保存初始梯度到 grad_old
    memcpy(grad_old, swe->a_z, swe->n * sizeof(double));

    FILE *fp_cost = fopen("cost_history.txt", "w");
    int max_iter = 80;

    for(int iter = 0; iter < max_iter; iter++) {
        printf("Iter %2d: Cost = %.6e ", iter, current_cost);
        fprintf(fp_cost, "%d %.10e\n", iter, current_cost); // 修复：添加分号
        
        if(current_cost < 1e-4) { 
            printf("\nConverged!\n"); 
            break; 
        }

        // --- A. 线搜索 (Line Search) ---
        double alpha = (iter == 0) ? 0.05 : 1.0;
        double c1 = 1e-4; // Armijo 参数
        double slope = dot_product(swe->a_z, direction, swe->n);
        
        int ls_iter = 0;
        double new_cost = 0.0;
        int success = 0;

        // 回溯循环
        while(ls_iter < 20) {
            // 试探更新: z_new = z_est + alpha * direction
            for(int i=0; i<swe->n; i++) {
                z_new_trial[i] = z_est[i] + alpha * direction[i];
            }

            new_cost = compute_cost(swe, z_new_trial, diff_z, obs_stride);

            // Armijo 条件检查
            if(new_cost <= current_cost + c1 * alpha * slope) {
                success = 1;
                break;
            }
            alpha *= 0.5;
            ls_iter++;
        }

        printf("| Alpha = %.4e | New Cost = %.6e\n", alpha, new_cost);

        // --- B. 更新状态 ---
        memcpy(z_est, z_new_trial, swe->n * sizeof(double));
        current_cost = new_cost;

        // --- C. 准备下一次迭代 (计算 beta) ---
        swe_run_adjoint(swe, diff_z, background_z);

        // 计算 Polak-Ribiere Beta
        double g_new_dot_g_new = dot_product(swe->a_z, swe->a_z, swe->n);
        double g_old_dot_g_old = dot_product(grad_old, grad_old, swe->n);
        double g_new_dot_diff  = 0.0;
        for(int i=0; i<swe->n; i++) {
            g_new_dot_diff += swe->a_z[i] * (swe->a_z[i] - grad_old[i]);
        }

        double beta = g_new_dot_diff / (g_old_dot_g_old + 1e-16);
        if (beta < 0) beta = 0;

        // 更新搜索方向
        for(int i=0; i<swe->n; i++) {
            direction[i] = -swe->a_z[i] + beta * direction[i];
        }

        // 保存当前梯度
        memcpy(grad_old, swe->a_z, swe->n * sizeof(double));
    }

    // 保存结果
    save_to_file("z_final_analysis.bin", z_est, swe->n);
    fclose(fp_cost);

    // 清理
    free1double(swe->h); free1double(swe->H); free1double(swe->z); 
    free1double(swe->u); free1double(swe->v); free1double(swe->a_z); 
    free1double(swe->a_u); free1double(swe->a_v); free1double(swe->a_H);
    free1double(swe->z_store); free1double(swe->u_store); free1double(swe->v_store);
    free1double(swe->z_obs); free1double(diff_z); 
    free1double(z_est); free1double(z_temp); free1double(u_temp); free1double(v_temp);
    free1double(z_true_init); free1double(grad_old); free1double(direction); free1double(z_new_trial);
    free(swe);

    return 0;
}
