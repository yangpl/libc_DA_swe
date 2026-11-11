#ifndef SWE_PRIMAL_H
#define SWE_PRIMAL_H
void swe_flux(int axis, const double left[3], const double right[3], double flux[3]);
void swe_euler(int n, int nx, int ny, double dx, double dy, double dt,
               const double input[3*n], double output[3*n]);
void swe_rk2(int n, int nx, int ny, double dx, double dy, double dt,
             double state[3*n], double stage[3*n], double tmp[3*n]);
#endif
