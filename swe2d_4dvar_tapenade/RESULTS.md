# Results from the executed experiments

## Tapenade C adjoint with native binomial checkpointing (new run)

The production observation-misfit adjoint is now generated from C by Tapenade.
The 60×60 experiment was rerun with 8 binomial snapshot slots; the earlier
results below remain the archived manual-adjoint reference.

- Converged by relative cost reduction after **309 iterations / 345 evaluations**.
- Cost: **11809.436263851 → 276.464216016**; gradient infinity norm **2.42925e-5**.
  The gradient tolerance of 1e-5 was not the stopping criterion.
- Window depth RMSE: **0.02114487 → 0.002634356** (87.5% reduction).
- Forecast depth RMSE: **0.02029201 → 0.0005183793** (97.4% reduction).
- Held-out sensor residual RMS: **0.02044503 → 0.003494142** (82.9% reduction).
- Full-grid gradients match the frozen manual adjoint to **3.10e-16** relative
  L2 error at the background and **1.69e-15** at the archived analysis.
- Eight Python tests, native oracle/finite-difference tests (including multi-block
  tapes), and ASan/UBSan pass. Three full-objective directional checks have best
  relative errors **3.31e-10, 1.29e-10, 1.49e-9**.

| Snapshot budget | Peak snapshots | Scheduled forward steps | Snapshot payload MB |
|---:|---:|---:|---:|
| 1 | 1 | 45150 | 0.259212 |
| 2 | 2 | 4900 | 0.518424 |
| 4 | 4 | 1938 | 1.036848 |
| 8 | 8 | 1280 | 2.073696 |
| 16 | 16 | 1029 | 4.147392 |

All budgets reverse 300 steps and return identical gradients. Counts exclude
the initial validation pass and local Euler replay; payload excludes local tape,
work arrays, and allocation overhead. This is Tapenade's own binomial scheduler.
The generated backend is slower than the specialized manual oracle in the
recorded timing samples; its advantages are automatic derivative generation
and a configurable storage/recomputation budget.

New results: [summary](var_output_tapenade/summary.json),
[assimilation plot](var_output_tapenade/var_4d_metrics.png),
[checkpoint plot](var_output_tapenade/checkpoint_tradeoff.png), and
[measured counts](var_output_tapenade/checkpoint_benchmark.csv).
Generation and the necessary checked C-backend/runtime repairs are documented
in [README.md](README.md#tapenade-generation-and-c-repairs).

## Archived manual-adjoint results

The original code was partly working: its forward model and wet-state adjoint
were consistent, and assimilation reduced some trajectory errors. Its optimizer
was faulty and stopped far from stationarity. The revised implementation passes
numerical tests and converges. The large reconstruction gains below are from a
new, explicitly specified twin experiment; they are not an apples-to-apples
comparison with the original sparse, noisier experiment.

## Controlled comparison on the original problem

The original baseline was compiled and run before cleanup; its unused source and
data directory have since been removed. For that historical comparison, observations
were reconstructed from the original random-number generator.
Both optimizers used the same 60 × 60 grid, 16 depth sensors, 30 observation
times, dt = 0.002, σobs = 0.01, and diagonal background covariance.

| Metric | Original optimizer | Revised optimizer |
|---|---:|---:|
| Initial cost | 258.765501 | 258.765501 |
| Final cost | 252.304637 | 249.700807 |
| Final scaled gradient infinity norm | 2.56745 | 1.79 × 10⁻⁶ |
| Iterations | 12, fixed limit | 16, converged |
| Initial depth RMSE | 0.0197433 | 0.0197420 |
| Space-time depth RMSE, including t = 0 | 0.00367623 | 0.00360609 |

The revised face-based forward solver agrees with the archived original truth
trajectory within **6.61 × 10⁻¹¹**, including CSV rounding. This isolates the
optimizer correction from a change in model dynamics.

The original experiment cannot recover most of its cellwise initial noise:
only 16 locations are observed, the observation noise is large relative to
later depth errors, and numerical diffusion erases small-scale information.
Converging the optimizer does not fix that information deficit. The converged
solution has lower aggregate window error, but the original early-stopped
solution has lower error at some late times. Minimizing a noisy observation
objective does not guarantee smaller truth error at every time.

See [controlled comparison plot](var_output/baseline_comparison.png) and
the historical table above (the baseline data directory has been removed).

## Revised 60 × 60 demonstration

Configuration: unit square, nonlinear Gaussian depth perturbation, 36 depth
sensors, observations every 0.02 from t = 0.02 to 0.30, dt = 0.001, independent
forecast from t = 0.32 to 0.50. Observation noise standard deviation is 0.003.
The background error is a random draw from the documented correlated covariance
(square-root smoothing length 0.06; spatially averaged σh = 0.03 and σmomentum
= 0.015). Seed: 20261016. Truth is used only to construct and evaluate this
synthetic experiment, never in the optimization objective.

The optimizer converged in **305 iterations**, with cost
**11809.4363 → 276.4642** and final scaled gradient infinity norm
**1.49 × 10⁻⁵**. SciPy's stopping reason was relative objective reduction;
the gradient was also close to its 10⁻⁵ tolerance. No invalid trial was accepted.

| Diagnostic | Background | Analysis | Reduction |
|---|---:|---:|---:|
| Initial depth RMSE | 0.0317399 | 0.0086690 | 72.7% |
| Assimilation-window depth RMSE | 0.0211449 | 0.00263436 | 87.5% |
| Assimilation-window hu RMSE | 0.0322686 | 0.00890184 | 72.4% |
| Assimilation-window hv RMSE | 0.0423474 | 0.00827857 | 80.5% |
| Independent forecast depth RMSE | 0.0202920 | 0.000518382 | 97.4% |
| Independent forecast hu RMSE | 0.0516189 | 0.00559197 | 89.2% |
| Independent forecast hv RMSE | 0.0550445 | 0.00541035 | 90.2% |
| Noisy observations at unused sensors: residual RMS | 0.0204450 | 0.00349414 | 82.9% |

Window RMSE is the square root of the mean squared error over all cells and
saved times, including the initial state. Forecast RMSE uses only times strictly
after the last assimilated observation. Unused-sensor observations have their
own noise and never enter the objective. Training residual RMS is about 0.002784,
close to the specified noise standard deviation of 0.003.

This validates this synthetic setup, not real-data performance. Truth and model
use the same discretization; covariance is matched to the generated background
errors. The open boundaries let the initial wave leave the domain, so later
forecast skill includes correction of persistent spurious background offsets.
The model is first order in space and is not intended for wetting/drying or
bathymetry without further work.

## Verification

* Six Python numerical regression tests pass: rest/moving uniform flow, invalid
  states and CFL rejection, multitime finite differences for every state variable
  including all corners, checkpoint interval equivalence, the covariance transpose
  and full objective gradient, and repeated-observation accumulation.
* Direct C tests pass: all three conservation balances match open-boundary fluxes;
  full-state Euler transpose relative error is 7.51 × 10⁻¹⁰; checkpointed
  observation-adjoint directional error is 3.05 × 10⁻¹⁰.
* GCC AddressSanitizer and UndefinedBehaviorSanitizer report no errors in the
  direct kernel and checkpointed-adjoint tests.
* Three full-experiment signed central-difference checks have best relative errors
  3.31 × 10⁻¹⁰, 1.29 × 10⁻¹⁰, and 1.49 × 10⁻⁹. Taylor remainders have fitted
  orders 1.9997, 1.9996, and 1.9993.
* All reported trajectories remain wet. Maximum checked stage CFL is below 0.414,
  below the enforced limit of 0.9. No clipping is used.
* On the original 600-step schedule, checkpointed adjoint storage is approximately
  6.7 MB versus 104 MB for full state/stage histories, excluding other outputs.

## Figures and saved data

* [Main diagnostics](var_output/var_4d_metrics.png)
* [Truth, background, analysis, and errors](var_output/var_4d_snapshots.png)
* [All three state-component errors](var_output/all_components_rmse.png)
* [Gradient and Taylor tests](var_output/fd_adjoint_check.png)
* [Original-problem comparison](var_output/baseline_comparison.png)
* [Animation](var_output/var_4d_result.gif)
* [Machine-readable summary](var_output/summary.json)
* [Saved trajectories and observations](var_output/experiment.npz)

Each static figure also has a PDF version. Reproduce the complete demonstration
with `bash run.sh`; see `README.md` for current parameters. The removed baseline experiment is
retained only through its previously saved tables and figures.
