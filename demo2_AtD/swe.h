#ifndef SWE_H
#define SWE_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define G 9.81
#define idx(i,j) ((i) + swe->nx*(j))

typedef struct {
    int nx, ny, nt;
    int verb;
    double dx, dy, dt;
    double tol, gamma, f;
    int niter;

    // 物理场
    double *h; // 地形
    double *H; // 总水深
    double *z; // 水位
    double *u; // 纬向速度
    double *v; // 经向速度

    // 伴随变量
    double *a_z;
    double *a_u;
    double *a_v;
    double *a_H; // 临时变量

    // 轨迹存储 (用于伴随模式)
    // 大小为 nt * nx * ny
    double *z_store;
    double *u_store;
    double *v_store;
    
    // 观测数据
    double *z_obs; 

    int n; // nx * ny
} swe_t;

// 内存管理工具
double* alloc1double(int n);
void free1double(double* p);

// 核心算子
void swe_step_forward(swe_t *swe, double *z_out, double *u_out, double *v_out);
void swe_run_adjoint(swe_t *swe, double *diff_z, double *background_z); // diff_z = z_model - z_obs
// 在 swe.h 底部添加
double dot_product(double *v1, double *v2, int n);
#endif
