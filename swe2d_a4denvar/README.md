# Analytic shallow-water 4D-EnVar (A-4DEnVar)

A standalone analytic solution of the fixed-ensemble, strong-constraint 4D-EnVar
quadratic. The nonlinear SWE model, ensemble construction, background covariance,
and twin-experiment setup are reused from `swe2d_4denvar`. The new solver explicitly
provides equivalent ensemble-space and observation-space analytic updates.

The Cholesky option in `swe2d_4denvar` already computes this analytic minimizer
in ensemble space. A-4DEnVar here solves the **same mathematical problem**;
it makes the analytic formulation and the two solve spaces explicit. It should
agree with converged iterative 4D-EnVar for identical ensembles and observations.

## Run

Requires a C11 compiler, Make, Python 3, and the packages in `requirements.txt`.
The normal build, tests, and assimilation need no sibling project or Tapenade.

```bash
cd swe2d_a4denvar
python3 -m pip install -r requirements.txt
bash run.sh
```

This builds, tests, assimilates, forecasts, and writes PNG/PDF plots. Defaults
match `swe2d_4denvar`: 60 × 60 cells, 80 members, 36 height sensors, 15 observation
times over 300 steps, then 200 free-forecast steps. Outputs go to
`a4denvar_output/`. Existing files in that output directory are overwritten.

Choose the factorization space:

```bash
python3 assimilate.py --solve-space auto --output a4denvar_output_auto
python3 assimilate.py --solve-space ensemble --output a4denvar_output_ensemble
python3 assimilate.py --solve-space observation --output a4denvar_output_observation
```

`auto` factors the smaller system, choosing ensemble space on a tie. The default
problem has 80 coefficients and 540 stacked observations, so it selects an
80 × 80 ensemble matrix. Both explicit options remain available for comparisons.

A quick example with fewer observations than ensemble members:

```bash
bash run.sh --nx 12 --ny 10 --stride 5 --cycles 4 --forecast-cycles 2 \
  --sensors-side 3 --members 40 --output a4denvar_output_small
```

Here `auto` chooses the 36 × 36 observation-space system instead of the 40 × 40
ensemble-space system. `run.sh` resolves relative output paths from this project;
direct Python commands resolve them from the caller's working directory.

## Analytic update

Let `xb` be the initial background and `Ne >= 2` the number of ensemble members,
whose initial mean is `xb`. Define normalized initial anomalies and stacked
predicted-observation anomalies:

```text
X0[:,e] = (member[e] - xb) / sqrt(Ne-1)
Y[:,e]  = (G(member[e]) - mean_e G(member[e])) / sqrt(Ne-1)
d       = observations - G(xb)
```

`G` integrates the nonlinear model to every observation time and samples depth.
The innovation uses the nonlinear background trajectory. The ensemble mean
prediction centers `Y` and is generally different from `G(xb)`.

With one common coefficient vector `w` for the entire window:

```text
x0(w) = xb + X0 w
Je(w) = 0.5 wᵀw + 0.5 (Yw-d)ᵀ R⁻¹ (Yw-d)
```

There is no additional `Ne-1` factor in the prior. This normalization gives the
sample state covariance `Be = X0 X0ᵀ`. Define whitened arrays `Z = R⁻¹ᐟ² Y`
and `v = R⁻¹ᐟ² d`. The analytic solutions are:

**Ensemble space:**

```text
(I_Ne + Zᵀ Z) w = Zᵀ v
xa = xb + X0 w
```

**Observation space:**

```text
(I_m + Z Zᵀ) u = v
w = Zᵀ u
xa = xb + X0 w
```

The observation-space expression is equivalently

```text
xa = xb + X0 Yᵀ (R + Y Yᵀ)⁻¹ d.
```

Both forms use one Cholesky factorization followed by triangular solves. No
explicit inverse, dense state covariance, adjoint, or iterative optimizer is
formed or used. Positive observation variances and the identity term make the
factorized systems positive definite even for rank-deficient ensembles.

The code checks the solved-system residual and the coefficient-space optimality
residual. `summary.json` reports the requested and selected space, system
dimension, one factorization, and zero optimization iterations. The two rows in
`cost_history.csv` represent the background and the analytic solution.

For `m` stacked observations, the matrix storage is O(Ne²) in ensemble space or
O(m²) in observation space, besides the anomalies. No repeated SWE integrations
are needed during the analytic solve. Members are propagated sequentially to
construct `Y`, retaining their predicted observations rather than every member's
full trajectory.

## Scope and nonlinear validation

The analytic result is exact, up to numerical roundoff, for the **fixed-ensemble
quadratic**. It is not a one-step exact solution of nonlinear full-state 4D-Var.
The initial increment is restricted to the ensemble span, with rank at most
`Ne-1`, and ensemble observation regression approximates nonlinear responses.
There are no nonlinear outer loops, localization, hybrid prior, or posterior
ensemble transformation in this implementation.

The recovered initial state is integrated again with the nonlinear SWE model
through the assimilation and forecast intervals. Saved `analysis` arrays are
that actual trajectory. Both the minimized surrogate cost and the true nonlinear
cost with the same coefficient prior are saved:

```text
Jnl(w) = 0.5 wᵀw + 0.5 ||R⁻¹ᐟ² (G(xb + X0 w) - observations)||².
```

Quadratic gradient checks do not claim to check the nonlinear objective's
derivative. Held-out observations and free-forecast error quantify the actual
nonlinear result; nonlinear improvement is not guaranteed by the analytic solve.

Observation errors are diagonal: the experiment uses scalar `--obs-std` and the
API also supports standard deviations broadcastable to `(cycles, nobs)`. Full
correlated observation error is not implemented. `--spread` multiplies initial
ensemble perturbations and changes the prior covariance by its square.

The reused C model evolves cell-centered `(h, hu, hv)` with Rusanov fluxes,
SSP-RK2, gravity 9.81, and zero-gradient open boundaries. It rejects nonfinite
states, depth at or below `1e-6`, and stage CFL above 0.9. There is no clipping,
bathymetry, forcing, or wetting/drying. Invalid members fail explicitly.

## Python API

```python
import numpy as np
from swe import Model
from covariance import BackgroundTransform
from aenvar import EnsembleProblem, make_ensemble

model = Model(nx=12, ny=10, stride=5, cycles=4)
xb = np.zeros(model.shape)
xb[0] = 1.
members = make_ensemble(xb, BackgroundTransform(model.shape), size=20, seed=17)
receivers = np.array([[.237, .419], [.681, .752]])
observations = np.full((4, 2), 1.001)
problem = EnsembleProblem(model, xb, members, observations, .003, receivers=receivers)
w, info = problem.objective.solve('auto')
analysis, diagnostics = model.forward(problem.initial_state(w))
```

Integer cell indices can replace physical receivers using `indices=...`.
Receiver sampling is bilinear at cell centers with constant boundary extension.
Observations exclude time zero; trajectories include it.

## Test drivers and files

```bash
make test                     # Python and native C numerical tests
make sanitize                 # ASan/UBSan native tests
make compare                  # Optional sibling manual forward-model comparison
python3 tests/check_saved.py --output a4denvar_output
python3 tests/compare_envar.py --output a4denvar_output
python3 plot_result.py --output a4denvar_output
```

The last comparison driver optionally imports the current sibling
`swe2d_4denvar/envar.py`, comparing its Cholesky and CG solutions on exactly the
same saved anomalies, innovations, and observation errors. It changes no sibling
files. All other numerical tests except `make compare` are standalone.

Tests cover an independent observation-space covariance update, augmented
least-squares solutions, both analytic spaces, automatic dimension selection,
normalization, finite differences, rank deficiency, permutations, repeated
observations, nonlinear/background centering, receivers, and complete saved
experiments. Native tests check conservation, uniform flow, and invalid states.

| File | Purpose |
|---|---|
| `aenvar.py` | Analytic solver, objective, ensemble construction |
| `swe_model.c/.h`, `swe_primal.c/.h`, `swe.py` | Standalone forward model |
| `covariance.py` | Reused DCT background covariance square root |
| `assimilate.py`, `run.sh` | Twin experiment and full workflow |
| `plot_result.py` | Nonlinear trajectory diagnostics and plots |
| `tests/` | Numerical tests and comparison drivers |

Saved outputs include `summary.json` with configuration and source hashes,
`experiment.npz` with ensemble and trajectory data, three CSV diagnostic files,
and `a4denvar_metrics` / `a4denvar_snapshots` as PNG/PDF. Saved-run checks add
`analytic_check.json` and `envar_comparison.json`. See [RESULTS.md](RESULTS.md)
for executed validation. `make clean` preserves experiment outputs.
