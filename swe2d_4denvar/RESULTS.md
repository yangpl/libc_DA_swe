# Executed validation, 2026-11-11

The standalone implementation passed `make test` (14 Python tests plus the native
C driver), `make sanitize` (ASan/UBSan), and `make compare`.

The independent covariance test checks both Cholesky and CG against
`B H.T solve(H B H.T + R, innovation)`, where `B` is computed independently with
`np.cov`. Other checks cover the ensemble quadratic gradient and Hessian,
normalization, rank-deficient ensembles, nonlinear/background centering,
off-grid and repeated receivers, and saved nonlinear experiment outputs.

The forward-only C model produced exactly identical trajectory values and
diagnostics to the current hand-coded 4D-Var forward model for:

| Grid | Time steps |
|---|---:|
| 2 × 2 | 1 |
| 9 × 7 | 35 |
| 60 × 60 | 500 |

## Default twin experiment

Configuration: 60 × 60 grid, dt = 0.001, observation stride 20, 15 observation
times, 10 subsequent forecast times, 36 height sensors, 80 ensemble members,
spread factor 1, background seed 20261016, ensemble seed 20261019, and default
DCT prior parameters. The ensemble initial-anomaly rank was 79. One background
and 80 ensemble integrations constructed the assimilation quadratic; truth,
background, and analysis diagnostics used additional full forecast integrations.

| Diagnostic | Value |
|---|---:|
| Background cost, also the initial quadratic cost | 11809.436263850512 |
| Minimized ensemble quadratic cost | 405.7627664313165 |
| Actual nonlinear analysis cost, same coefficient prior | 422.2917984444327 |
| Final quadratic gradient infinity norm | 2.2744e-12 |
| Cholesky normalized equation residual | 4.8948e-16 |
| Nonlinear versus surrogate prediction RMS difference | 8.7681e-4 |
| Background forecast depth RMSE | 0.02029201156 |
| Analysis forecast depth RMSE | 0.001425192914 |
| Forecast depth RMSE reduction | 92.98% |
| Background held-out observation residual RMS | 0.02044503155 |
| Analysis held-out observation residual RMS | 0.004586584590 |

The three signed finite-difference checks of the quadratic objective had best
relative errors 1.26e-16, 1.39e-15, and 6.93e-15. These check the ensemble-space
derivative, not the derivative of the nonlinear SWE objective.

`tests/check_saved.py` independently reconstructed the quadratic from saved
member predictions. CG converged in 148 iterations with normalized equation
residual 9.20e-13. Its coefficients differed from Cholesky by at most 9.95e-10
(relative L2 difference 2.44e-10). Reintegrating the saved analysis initial state
reproduced the saved nonlinear trajectory exactly.

Files: `envar_output/summary.json`, `experiment.npz`, `solver_check.json`, CSV
diagnostics, and `envar_metrics` / `envar_snapshots` as PNG and PDF.

## Complete CG workflow

The documented `bash run.sh` small example also passed from build/tests through
assimilation and plotting: 12 × 10 cells, 20 members, 9 sensors, stride 5,
4 assimilation observation times, 2 forecast times. CG converged in 30 iterations
with normalized equation residual 2.74e-13. The quadratic cost decreased from
1429.52842 to 26.84294; actual nonlinear analysis cost was 27.08521. Forecast
depth RMSE decreased from 0.0198661 to 0.0176029 (11.39%).

Saved results are in `envar_output_small/`. The different improvement from the
default example reflects its different grid, window, and ensemble size. A
finite-rank 4D-EnVar approximation is not expected to reproduce full-state
nonlinear 4D-Var exactly. Experiment summaries record source hashes and package
versions for reproducibility.
