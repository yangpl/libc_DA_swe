/* rng.c — pseudo-random number generation (LCG + Box–Muller).
 * Same generator as the swe2d_enkf framework. */

#include <math.h>

#include "rng.h"

#define PI 3.14159265358979323846

void rng_seed(Rng *rng, unsigned long long seed) {
  rng->state = seed ? seed : 1ULL;
  rng->has_spare = 0;
  rng->spare = 0.0;
}

double rng_uniform01(Rng *rng) {
  rng->state = rng->state * 2862933555777941757ULL + 3037000493ULL;
  return ((double)(rng->state >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

double rng_normal(Rng *rng) {
  if (rng->has_spare) {
    rng->has_spare = 0;
    return rng->spare;
  }
  {
    double u1 = rng_uniform01(rng);
    double u2 = rng_uniform01(rng);
    double r = sqrt(-2.0 * log(u1));
    double th = 2.0 * PI * u2;
    rng->spare = r * sin(th);
    rng->has_spare = 1;
    return r * cos(th);
  }
}