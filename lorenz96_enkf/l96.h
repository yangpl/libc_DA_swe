#ifndef L96_H
#define L96_H

/* Single RK4 step of the Lorenz-96 model.  scratch must hold 5*n doubles. */
void l96_step_rk4(double *x, int n, double F, double dt, double *scratch);

/* Integrate forward by 'duration' time units (steps of dt). */
int l96_model_run(double *x, int n, double F, double dt,
                  double duration, double *scratch);

/* Advance all Ne ensemble members. */
int advance_ensemble(double *ens, int Ne, int n, double F, double dt,
                     double duration, double *scratch);

#endif