#ifndef LINALG_H
#define LINALG_H

/* Dense Gauss–Jordan matrix inverse (returns 1 on success). */
int invert_matrix(double *A, double *Ainv, int m);

/* Symmetric Jacobi eigendecomposition A = V Lambda V^T (returns 1 on success). */
int symmetric_eigen_jacobi(const double *A, double *V, double *eval, int n);

/* Gaspari-Cohn (1999) fifth-order correlation function.
 * d = distance in grid points; lam = half-width (0 => returns 1.0). */
double gc_corr(double d, double lam);

/* Minimum cyclic distance on a ring of length n. */
double cycl_dist(int i, int j, int n);

#endif