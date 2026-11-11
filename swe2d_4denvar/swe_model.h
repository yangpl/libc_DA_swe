#ifndef SWE_MODEL_H
#define SWE_MODEL_H
/* States: float64 [3][ny][nx], components (h, hu, hv).
 * Output trajectory: [nsteps/stride+1][3][ny][nx], including initial state.
 * Diagnostics: minimum depth and maximum 2-D CFL over every RK/Euler stage.
 * Return codes: 0 success; 1 invalid arguments; 2 allocation failure;
 * 3 nonfinite/dry state; 4 CFL > 0.9; 5 nonfinite sampled value.
 * No global mutable state. Input/output arrays must not overlap. */
int swe_forward(const double *initial, int nx, int ny, double lx, double ly,
                double dt, int nsteps, int stride, double *trajectory, double *diagnostics);
/* Bilinear sampling at fixed physical coordinates in [0,lx] x [0,ly].
 * Values live at cell centres; boundary half-cells use constant extension. */
int swe_sample_field(const double *values, int nx, int ny, double lx, double ly,
                     int nobs, const double *rx, const double *ry, double *samples);
#endif
