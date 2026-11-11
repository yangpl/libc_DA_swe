# Executed analytic 4D-EnVar validation, 2026-11-11

`make test` passed all 17 Python tests and the native C driver. `make sanitize`
passed ASan/UBSan checks. `make compare` reproduced the current hand-coded
4D-Var forward trajectories and diagnostics exactly on 2 × 2 / 1-step,
9 × 7 / 35-step, and 60 × 60 / 500-step cases.

The analytic tests include independent augmented least-squares and covariance
oracles, both solve spaces, automatic dimension selection, finite differences,
rank deficiency, zero innovation, member/observation permutations, repeated
observations, ensemble normalization, and nonlinear/background centering.

## Default 60 × 60 experiment

The complete `bash run.sh` workflow passed with 80 members, 36 sensors, 15
observation times, 300 assimilation steps, and 200 forecast steps. Automatic
selection used the 80 × 80 ensemble system instead of the 540 × 540 observation
system. The analytic solve used one factorization and zero optimization iterations.

| Quantity | Result |
|---|---:|
| Initial quadratic cost | 11809.436263850512 |
| Final quadratic cost | 405.7627664313165 |
| Actual nonlinear analysis cost | 422.2917984444327 |
| Ensemble-form optimality residual / max(1, RHS norm) | 4.8948e-16 |
| Observation-form optimality residual / max(1, RHS norm) | 1.2719e-14 |
| Maximum coefficient difference between analytic forms | 3.4794e-13 |
| Relative L2 coefficient difference | 9.5659e-14 |
| Maximum initial-state difference between analytic forms | 6.9250e-15 |
| Background forecast depth RMSE | 0.02029201156 |
| Analysis forecast depth RMSE | 0.001425192914 |
| Forecast depth RMSE reduction | 92.98% |

`tests/check_saved.py` rebuilt the objective from saved ensemble predictions,
checked both solve spaces, and reintegrated the recovered initial state. The
reintegrated nonlinear trajectory matched the saved analysis exactly.

## Agreement with existing 4D-EnVar

`tests/compare_envar.py` imported the current sibling `swe2d_4denvar/envar.py`
and used exactly the same anomalies, innovation, and observation errors.

- The analytic ensemble-space coefficients exactly matched the sibling Cholesky
  solution.
- The sibling CG solve took 148 iterations. Its maximum coefficient difference
  from the analytic result was 9.9411e-10, or 2.4346e-10 relative L2. The maximum
  corresponding initial-state difference was 2.0087e-11.
- Comparing the fresh default experiment against the previously saved default
  `swe2d_4denvar/envar_output/experiment.npz` gave identical values for every
  saved array, including members, observations, controls, and analysis forecasts.

This agreement is expected: the old Cholesky option already solves the same
fixed-ensemble quadratic analytically. The new project makes both analytic
forms explicit; it does not change the prior or the assimilation objective.

## Automatic observation-space example

The documented small `bash run.sh` example also completed tests, assimilation,
forecast, and plotting: 12 × 10 cells, 40 members, 9 sensors, four observation
times, stride 5, and two subsequent forecast times. Automatic selection used a
36 × 36 observation system instead of a 40 × 40 ensemble system.

The quadratic cost decreased from 1429.52842 to 21.17055159; the actual nonlinear
analysis cost was 21.35318446. The maximum initial-state difference between the
two analytic forms was 1.2212e-15. Forecast depth RMSE decreased from 0.01986609
to 0.01582655. Momentum errors did not both improve in this smaller example;
minimizing the ensemble quadratic does not guarantee improvement in every
physical variable.

## Saved artifacts

`a4denvar_output/` and `a4denvar_output_small/` contain summaries, complete saved
experiment arrays, diagnostic CSV files, and PNG/PDF plots. `analytic_check.json`
records the two-space verification; `envar_comparison.json` records the sibling
solver comparison for the default run. Summaries include source hashes and
Python/NumPy/SciPy versions. Validation and run logs are saved with the outputs.

The analytic solution is exact up to roundoff for the fixed-ensemble quadratic,
not for the original nonlinear full-state 4D-Var problem. Surrogate and actual
nonlinear costs are reported separately throughout.
