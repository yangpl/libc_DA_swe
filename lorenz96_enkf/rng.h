#ifndef RNG_H
#define RNG_H

typedef struct {
  unsigned long long state;
  int has_spare;
  double spare;
} Rng;

void rng_seed(Rng *rng, unsigned long long seed);
double rng_uniform01(Rng *rng);
double rng_normal(Rng *rng);

#endif