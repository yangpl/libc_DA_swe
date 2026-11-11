# Shallow-water 4D-EnVar

A standalone, adjoint-free, strong-constraint initial-state assimilation example.
The forward model and twin-experiment configuration are reused from the existing
`swe2d_4dvar_*` projects. C advances the nonlinear shallow-water equations;
Python constructs the ensemble and solves one joint assimilation window in
ensemble coefficient space. All times share the same coefficients.

See [RESULTS.md](RESULTS.md) for the executed numerical checks and example results.

## Run

Requires a C11 compiler, Make, Python 3, and the packages in `requirements.txt`.
No Tapenade installation or sibling directory is needed for the normal build,
tests, or assimilation.

```bash
cd swe2d_4denvar
python3 -m pip install -r requirements.txt
bash run.sh
```

`run.sh` builds, runs the tests, assimilates, forecasts, and creates PNG/PDF plots.
The default experiment uses 60 × 60 cells, 80 ensemble members, 36 depth sensors,
300 assimilation steps, and 200 forecast steps. Results go to `envar_output/`.
Existing files in the selected output directory are overwritten. Use a new
`--output` path to preserve an earlier experiment.

Quick rectangular-grid demonstration using the iterative solver:

```bash
bash run.sh --nx 12 --ny 10 --stride 5 --cycles 4 --forecast-cycles 2 \
  --sensors-side 3 --members 20 --solver cg --output envar_output_small
```

Individual stages and test drivers:

```bash
make                         # Standalone forward library only
make test                    # Python numerical tests and native C driver
make sanitize                # C address/undefined-behavior sanitizers
python3 assimilate.py --help
python3 assimilate.py --members 120 --output envar_output_120
python3 plot_result.py --output envar_output_120
python3 tests/check_saved.py --output envar_output_120
make compare                 # Optional: compare against the sibling manual model
```

`run.sh` resolves relative output paths from this directory. Direct Python
commands resolve them from the caller's current directory. `make clean` removes
only build products, leaving experiments intact.

## Fixed-ensemble formulation

Let `Ne >= 2` and let initial ensemble members have mean `xb`. The columns of
`X0` are `(member[e] - xb)/sqrt(Ne-1)`, so the sample prior is `Be = X0 X0.T`.
The ensemble uses independent, spatially correlated draws from the same DCT
covariance square root as the 4D-Var example, centered across members.
`--spread` multiplies perturbations and therefore multiplies the sample
covariance by its square. Background error and ensemble perturbations use
separate reproducible random seeds.

Propagate the background and each member with the full nonlinear model. At each
observation time, center the member predictions around their ensemble mean and
divide by `sqrt(Ne-1)`. Stack these observation anomalies into `Y`, and define
the innovation using the **nonlinear background trajectory**:

```text
d = observations - G(xb)
initial(w) = xb + X0 w
G(initial(w)) ≈ G(xb) + Y w

Je(w) = 0.5 wᵀw + 0.5 ||R^(-1/2) (Y w - d)||²
∇Je(w) = w + Yᵀ R^(-1) (Y w - d)
A = I + Yᵀ R^(-1) Y
A w = Yᵀ R^(-1) d
```

This follows the background-trajectory convention in the repository's 4D-EnVar
notes. For nonlinear dynamics, the mean ensemble observation generally differs
from `G(xb)`; it is used to center `Y`, not to replace the innovation center.
The covariance normalization occurs once. There is no extra `Ne-1` factor in
the coefficient prior.

`--solver cholesky` solves this positive-definite ensemble system directly,
without forming an inverse. `--solver cg` uses conjugate gradients and products
`p + Yw.T @ (Yw @ p)` with whitened `Yw`; it does not form the Hessian. Both solve
the same quadratic problem. Singular sample covariance and the redundant
ensemble-mean direction do not make `A` singular because of the identity prior.
After the ensemble integrations, neither solver calls the forward model.

The observation covariance is diagonal. The experiment uses a single positive
`--obs-std`; the Python API also accepts positive standard deviations broadcastable
to `(cycles, nobs)`. Full correlated observation covariance is not implemented.

## Relationship to nonlinear 4D-Var

This implementation is a **single fixed-ensemble quadratic approximation**.
It does not implement nonlinear outer loops, localization, a hybrid full-rank
prior, inflation adaptation, or a posterior ensemble transformation.

With `Ne` members, admissible initial increments have rank at most `Ne-1`.
The sample covariance differs from the full covariance used by 4D-Var, and the
ensemble regression only approximates nonlinear observation responses.
Consequently, matching the 4D-Var experiment settings does **not** imply an
identical analysis. Increasing the ensemble size reduces sampling limitations
but does not make the finite-amplitude nonlinear approximation exact.

The recovered initial state is always integrated again through the nonlinear
model for the entire assimilation and free-forecast interval. Saved `analysis`
arrays are this real trajectory, not a weighted sum of member trajectories.
Diagnostics report both the minimized surrogate cost and

```text
Jnl(w) = 0.5 wᵀw + 0.5 ||(G(xb + X0 w) - observations)/obs_std||².
```

The solver's gradient norm and gradient checks apply to `Je`, not `Jnl`.
Convergence of the quadratic solve does not guarantee nonlinear-cost reduction
or improved truth error. `nonlinear_cost_reduced`, held-out residuals, and free
forecast RMSE make these distinctions explicit. An invalid ensemble member or
nonlinear analysis fails visibly; the code never clips depth or silently changes
the ensemble. Reduce the ensemble spread, background scales, or time step as
appropriate for a different experiment. `spread` changes the prior as well as
perturbation amplitude; it is not a finite-difference step size.

## Forward model and observations

The model uses cell-centered conservative `(h, hu, hv)` states, Rusanov fluxes,
SSP-RK2 time stepping, gravity 9.81, and zero-gradient open boundaries. Every
Euler/RK stage must have finite state, depth greater than `1e-6`, and 2-D CFL
no greater than 0.9. There is no bathymetry, forcing, or wetting/drying.

The C kernels and checked forward API were extracted from
`swe2d_4dvar_tapenade/swe_primal.c` and `swe2d_4dvar.c`, retaining their arithmetic
and removing all AD dependencies. The DCT background transform was copied from
`swe2d_4dvar_checkpointing/assimilate.py`. Original source files are not changed.

States have shape `(3, ny, nx)`; trajectories `(cycles+1, 3, ny, nx)` include
time zero. Observations have shape `(cycles, nobs)` and exclude time zero.
`Model.observe` accepts either integer cell indices or physical receivers.
Physical receivers use bilinear cell-center interpolation, with constant
extension in the boundary half-cells. Repeated receivers are supported.

```python
import numpy as np
from swe import Model
from covariance import BackgroundTransform
from envar import EnsembleProblem, make_ensemble

model = Model(nx=12, ny=10, stride=5, cycles=4)
xb = np.zeros(model.shape)
xb[0] = 1.0
members = make_ensemble(xb, BackgroundTransform(model.shape), size=20, seed=17)
receivers = np.array([[.237, .419], [.681, .752]])
observations = np.full((4, 2), 1.001)
problem = EnsembleProblem(model, xb, members, observations, .003, receivers=receivers)
w, info = problem.objective.solve('cholesky')
analysis, diagnostics = model.forward(problem.initial_state(w))
```

Members are propagated sequentially. Memory holds initial anomalies, member
observation predictions, and one forward trajectory at a time, rather than all
member trajectories. The C stepper uses four work states; it has no adjoint tape
or checkpoint schedule. Large ensemble sizes still increase initial-state
storage and linear algebra costs.

## Files and outputs

| File | Purpose |
|---|---|
| `swe_model.c/.h`, `swe_primal.c/.h` | Forward-only C model and receiver sampling |
| `swe.py` | Validated NumPy/ctypes interface |
| `covariance.py` | DCT prior square root |
| `envar.py` | Ensemble construction, quadratic objective, Cholesky/CG solves |
| `assimilate.py` | Twin experiment, nonlinear verification, diagnostics |
| `plot_result.py` | Headless metrics and snapshots |
| `tests/test_envar.py` | Dense covariance oracle, derivatives, centering, rank deficiency |
| `tests/test_swe.py` | Forward API, receivers, complete assimilation experiment |
| `tests/test_forward.c` | Native conservation, boundary, and validity tests |
| `tests/compare_forward.py` | Optional current sibling forward-model comparison |
| `tests/check_saved.py` | Re-solve saved experiments with both methods and verify outputs |

Each successful experiment saves `summary.json`, `experiment.npz`,
`cost_history.csv`, `gradient_check.csv`, and `trajectory_metrics.csv`.
`experiment.npz` includes initial members, member observation predictions,
coefficients, observations, and nonlinear truth/background/analysis trajectories.
The summary records configuration, source hashes, package versions, ensemble
rank, solver residual, both cost definitions, and validation metrics.
Plots are `envar_metrics.png/.pdf` and `envar_snapshots.png/.pdf`.

The numerical tests compare both solves against an independent observation-space
Kalman/covariance calculation, verify covariance normalization against `np.cov`,
check signed finite differences and the quadratic Taylor remainder, and exercise
all three state components and multitime observations. A nonlinear toy model
specifically detects mixing up background and ensemble-mean innovations.
