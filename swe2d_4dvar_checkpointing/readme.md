# Shallow-water 4D-Var with binomial checkpointing

A standalone two-dimensional shallow-water twin experiment. The C code evolves
water depth and momenta `(h, hu, hv)` and computes the exact discrete observation
gradient. Python constructs a correlated background covariance, optimizes the
full initial state with SciPy L-BFGS-B, and evaluates a subsequent free forecast.
The support scripts are adapted from `../swe2d_4dvar` for this directory's C API.
They use only files in this directory and need no Tapenade installation.

## Setup and running

Requires GCC (or another C11 compiler), GNU Make, Python 3, and these packages:

```bash
python3 -m pip install -r requirements.txt
bash run.sh
```

`run.sh` builds `libswe.so`, runs the numerical tests, performs the assimilation,
and saves figures and an animation in `var_output/`. It can be invoked from any
working directory. Relative output paths are resolved inside this project when
using `run.sh`; standalone Python commands resolve them from the current directory.
Existing files in the selected output directory are overwritten.

The default experiment uses 60 × 60 cells, 36 depth sensors, 300 assimilation
steps, and 200 free-forecast steps. For a quick rectangular-grid example:

```bash
bash run.sh --nx 12 --ny 10 --stride 5 --cycles 4 --forecast-cycles 2 \
  --sensors-side 3 --checkpoint-budget 8 --output var_output_binomial
```

Run individual stages or inspect all options:

```bash
make                         # Build only; CC and CFLAGS may be overridden
make test                    # Numerical Python and native C tests
make sanitize                # Native C address/undefined-behavior checks
python3 assimilate.py --help
python3 assimilate.py --output var_output_new
python3 plot_4dvar_result.py --output var_output_new --movie
make clean                   # Remove compiled products; preserve results
```

The Python interface also builds the library automatically with Make.
`--checkpoint-budget` sets the maximum number of in-memory state snapshots
(default 8, range 1..90). `--stride` sets only the observation interval; checkpoint
placement is independent of observation times. `--cycles` sets the number of
observation times in one joint assimilation window, not separate sequential analyses.
`--forecast-cycles` must be positive. Sensors must occupy distinct cells and
leave at least one cell for held-out validation.

## Files and outputs

| File | Purpose |
|---|---|
| `swe2d_4dvar.c`, `swe2d_4dvar.h` | Forward model and manual discrete adjoint |
| `swe.py` | Validated NumPy/ctypes interface and automatic build |
| `assimilate.py` | Synthetic truth and observations, covariance, optimization, diagnostics |
| `plot_4dvar_result.py` | Headless PNG/PDF figures and optional GIF from saved data |
| `run.sh`, `Makefile`, `requirements.txt` | Complete workflow, build/test targets, dependencies |
| `tests/` | Finite differences, binomial work bounds, original block-gradient comparisons, interface and conservation checks |
| `benchmark_checkpointing.py` | Storage, measured work, and timing comparison against the frozen block adjoint |
| `swe_ATD.tex`, `swe_ATD.pdf` | Historical mathematical notes; current scheduling is documented below |

Each experiment writes:

| Output | Contents |
|---|---|
| `experiment.npz` | Coordinates, sensors, observations, control, truth/background/analysis trajectories |
| `summary.json` | Configuration, convergence, RMSE, stage depth/CFL, checkpoint storage, source hashes |
| `cost_history.csv` | Initial and accepted-iteration costs and gradient norms |
| `trajectory_metrics.csv` | All-component RMSE and mass at each output time |
| `gradient_check.csv` | Signed directional finite differences and Taylor residuals |

Plotting writes `var_4d_metrics`, `var_4d_snapshots`, `all_components_rmse`, and
`fd_adjoint_check` in both PNG and PDF. `--movie` adds `var_4d_result.gif`.
Optimization failure saves its diagnostics and exits with an error; `run.sh`
then stops before plotting. Saved diagnostics can still be plotted separately.

## Model and interface

States are contiguous float64 arrays of shape `(3, ny, nx)` in component order
`(h, hu, hv)`. Saved trajectories have shape `(cycles + 1, 3, ny, nx)` and include
the initial state. Observations sample depth after each `stride` steps, excluding
time zero. Sensor index `i + nx*j` selects grid cell `(j, i)`.

```python
import numpy as np
from swe import Model

model = Model(nx=12, ny=10, stride=5, cycles=4, checkpoint_budget=8)
initial = np.zeros(model.shape)
initial[0] = 1.0
trajectory, diagnostics = model.forward(initial)
cost, gradient = model.evaluate(initial, [0, 119], np.ones((4, 2)), .003)
print(model.last_checkpoint_stats)
cost_only, _ = model.evaluate(initial, [0, 119], np.ones((4, 2)), .003,
                              gradient=False)
```

The observation cost and full initial-state gradient come from C. Python uses
`x0 = xb + L z` and adds the background penalty:

```text
J(z) = 0.5 zᵀz + 0.5 Σ_k ||(H M_k(xb + L z) - y_k) / σobs||²
∇z J = z + Lᵀ λ0
```

`L` is a symmetric Gaussian filter in an orthonormal cosine basis, with separate
depth and momentum scales. `--length 0` selects diagonal covariance. Synthetic
background perturbations and observation noise use reproducible random seeds.
Three directional finite-difference checks of the full objective run before
optimization. Training sensors, unused sensors, and an unassimilated forecast
provide separate error diagnostics.

The solver uses Rusanov finite-volume fluxes, SSP-RK2, and zero-gradient open
boundaries. It has no bathymetry, forcing, or wetting/drying. Invalid stages
(nonfinite values, depth ≤ 1e-6, or 2-D CFL > 0.9) raise `ModelError`; invalid
optimization trials are rejected without clipping. Reduce `--dt` for finer
grids if the CFL check fails. Total mass may change through the open boundaries.
This same-model synthetic experiment does not assess real-world model error.

## Binomial checkpointing

`swe_evaluate_binomial()` contains a self-contained adaptation of the binomial
placement and restore / redistribute / replay scheme by Pengliang Yang (2017).
The build, tests, sanitizer checks, and scripts require no external demo files.
Every checkpoint contains the complete `(h, hu, hv)` state at a scheduled time.
All slots are in memory; the demo's disk-storage tier is not needed here.

1. Advance once through the whole window, accumulate the observation cost,
   and save states only at binomial checkpoint times. Retain the input and first
   Euler stage of the final RK2 step.
2. Inject the final observation residual and reverse the final RK2 step.
3. For each earlier step, restore the latest checkpoint at or before its input
   time, redistribute available slots, and replay forward to that time. An exact
   checkpoint hit also restores the actual state payload.
4. Rebuild only the first Euler stage, then apply the existing exact RK2 adjoint.
   Inject an observation residual once when its physical time is reached in
   reverse. Replays add neither cost nor observation forcing.

The initial pass checks depth, finiteness, and CFL at every Euler/RK2 stage.
Deterministic replay uses identical arithmetic and omits those repeated validity
scans; diagnostics describe the fully validated original trajectory.

The schedule counts states including its base. Let `c` be the effective slot
count and `r` the smallest integer with `binomial(c+r, r) >= nsteps`. The full
RK2 forward count is `r*nsteps - binomial(c+r, r-1) + 1`, taking a binomial
coefficient with a negative lower argument as zero. The extra endpoint step
computes the final observation cost. There are also
`nsteps - 1` single-Euler stage reconstructions. These are reported separately
because they cost less than a complete RK2 step. The discrete adjoint kernels
and their derivative formulas are unchanged.

The implementation uses 64-bit binomial arithmetic, limits windows to 1,000,000
steps, handles one-step windows directly, and caps excess checkpoint budgets
at `max(1, nsteps - 1)`. Unused slots in shrinking replay intervals are discarded
from the schedule, avoiding duplicate checkpoint positions. All scheduling
and state storage belongs to the evaluation; there is no global mutable state.

### Memory and computation

For `N = nx*ny` cells and `c` effective checkpoint slots, the C gradient evaluation
allocates `(c + 6) * 3*N*8` bytes of state arrays. Six work arrays hold the current
state, RK2 stage, two scratch fields, and two adjoint fields. Two fixed-size
integer arrays hold the checkpoint schedule. Storage is independent of the
observation interval and window length for a fixed budget.

Cost-only evaluations stream through the observations with **four state arrays**,
no checkpoints, and no replay. `swe_forward()` still returns the requested
trajectory to its caller. Python truth/background/analysis trajectories,
optimizer memory, inputs/outputs, and allocator overhead are excluded from the
C state-storage statistics; these are not process peak-memory estimates.

For the default 60 × 60 grid and 300-step window, the former block implementation
used 62 state arrays (5.3568 MB). Eight binomial slots need 14 arrays (1.2096 MB),
a **77.4% reduction**. This is a memory/computation trade-off: a small budget can
require more forward work and take longer than the former block implementation.
Increasing the budget reduces replay at the cost of additional memory. The
scheduler optimizes the demo's state-only replay problem, not the total runtime
of the multistage RK2 adjoint.

Run a reproducible comparison on the same state and observations:

```bash
python3 benchmark_checkpointing.py --budgets 2 4 8 16 32 56 --repeat 3
```

It verifies every gradient against `tests/reference/block_swe.c`, measures median
wall times, and saves `var_output_checkpoint_benchmark/checkpoint_benchmark.csv`.
The frozen reference is used only for tests and benchmarks; production builds
compile only `swe2d_4dvar.c`.

Example measurement on this machine (60 × 60 cells, 300 steps; median of three
evaluations, all gradients identical to the frozen reference):

| Method / slots | C state arrays (MB) | Full RK2 forward steps | Extra Euler stages | Time (s) |
|---|---:|---:|---:|---:|
| Original block | 5.3568 | 600 | 0 | 0.4308 |
| Binomial / 4 | 0.8640 | 1639 | 299 | 0.6837 |
| Binomial / 8 | 1.2096 | 981 | 299 | 0.5061 |
| Binomial / 16 | 1.9008 | 730 | 299 | 0.4441 |
| Binomial / 32 | 3.2832 | 567 | 299 | 0.4040 |
| Binomial / 56 | 5.3568 | 543 | 299 | 0.4236 |

Times depend on the machine and workload. The default eight slots favor memory;
use a larger budget when reducing replay is more important.

### C API and diagnostics

The existing `swe_evaluate()` signature is preserved and now selects eight
binomial checkpoint slots. Call the explicit-budget API to choose a budget:

```c
SweCheckpointStats stats;
int status = swe_evaluate_binomial(initial, nx, ny, lx, ly, dt,
    nsteps, stride, nobs, indices, observations, obs_std, checkpoint_budget,
    &cost, gradient, diagnostics, &stats);
```

Use `gradient=NULL` for cost only and `stats=NULL` if counters are not needed.
Statistics describe one evaluation and include effective slots, allocated state
bytes, full forward/recomputed RK2 steps, stage reconstructions, reverse steps,
and checkpoint saves/restores. Counters are reset on entry and can be partial
on failure. Python exposes them as `model.last_checkpoint_stats`; an experiment's
`summary.json` records the last gradient evaluation's counters under `adjoint`.
The requested budget is also saved in the experiment configuration.

`make test` checks checkpoint placement invariants, actual step counts against
an independent binomial formula, and gradients against the original block implementation for many
window lengths and budgets, repeated sensors, observations at every step,
only-final observations, surplus slots, single-step windows, and arithmetic near
the million-step limit. `make sanitize` exercises the native tests with address
and undefined-behavior sanitizers.
