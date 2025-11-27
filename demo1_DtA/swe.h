#ifndef SWE_H
#define SWE_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define G 9.81

typedef struct {
    int nx, ny, nt;
    int verb;
    float dx, dy, dt;
    float tol, gamma, f;
    int niter;

    float *h; // 地形

    //forward fields
    float *H; // 总水深
    float *z; // 水位
    float *u; // 纬向速度
    float *v; // 经向速度

    //adjoint fields
    float *a_z;
    float *a_u;
    float *a_v;
    float *a_H; // 临时变量

    // 轨迹存储 (用于伴随模式), size=nt * nx * ny
    float *z_store;
    float *u_store;
    float *v_store;
    
    // 观测数据
  int stride_x, stride_y;
  int np;//np positions
  
  float *z_obs;//observed field z
  float *z_cal;//calculated field z
  float *z_res;//residual z_res=z_obs-z_cal

  float *z_temp;
  float *u_temp;
  float *v_temp;

    int n; // nx * ny
} swe_t;

#endif
