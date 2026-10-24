# Shallow-water 4D-Var

Strong-constraint 4D-Var for a two-dimensional shallow-water twin experiment.
The nonlinear solver and Tapenade-generated discrete adjoint run in C. Python
provides the background covariance, SciPy L-BFGS-B optimization, experiment
configuration, validation, and plotting. The controls are the complete initial
water-depth and momentum fields `(h, hu, hv)`.

The generated time loop uses Tapenade's native binomial checkpointing and calls
`adBinomial_next(&action, &step)` directly. Its snapshot budget is independent
of the observation interval.

## Setup and quick start

Requires GCC, GNU Make 4.3 or later, `patch`, Python 3, the packages in
[requirements.txt](requirements.txt), and the installed Tapenade runtime.
The tested toolkit is Tapenade 3.16. Java and the Tapenade generator are needed
only when regenerating the adjoint.

```bash
# Default toolkit location; change this if yours is installed elsewhere.
export TAPENADE_HOME="$HOME/Install/tapenade_3.16"
python3 -m pip install -r requirements.txt

# Build, test, assimilate, forecast, and plot into a new output directory.
bash run.sh --checkpoint-budget 8 --output new_run
```

The default experiment uses a 60 × 60 grid, 36 height sensors, 300 assimilation
steps, and a subsequent free forecast. A full run takes a few minutes on one
CPU core. `run.sh` saves PNG/PDF figures and a GIF after successful optimization.

Without `--output`, files go to `var_output/`. Running again with the same output
path overwrites that experiment's files. Use a new directory to preserve the
saved reference experiments.

For a smaller rectangular-grid example:

```bash
bash run.sh --nx 24 --ny 20 --cycles 8 --forecast-cycles 4 \
  --sensors-side 4 --checkpoint-budget 8 --output small_run
```

You can also run the stages separately:

```bash
make
make test
python3 assimilate.py --output new_run
python3 plot_4dvar_result.py --output new_run --movie
python3 assimilate.py --help
```

## Build, generation, and cleanup

| Command | Purpose |
|---|---|
| `make` | Prepare the runtime and build `libswe.so`; verify generated-source hashes |
| `make adjoint-generate` | Regenerate the adjoint from `swe_primal.c` using Tapenade |
| `make test` | Run Python regressions, C gradient/conservation tests, and scheduler tests |
| `make sanitize` | Check the C driver, generated adjoint, and runtime with ASan/UBSan |
| `make clean` | Remove current build objects, the library, test executables, and temporary runtime/generation files |

`make generate-adjoint` is an alias for `make adjoint-generate`. Regenerate
whenever the primal source or its header changes; ordinary builds reject stale
generated sources. The included generated adjoint can otherwise be compiled
without invoking Tapenade.

The runtime comes from `$TAPENADE_HOME/ADFirstAidKit`. The build copies the
required files and toolkit license into `build/tapenade_runtime/`, applies
[tapenade_alignment.patch](tapenade_alignment.patch), and compiles that copy.
The patch fixes four unaligned accesses on the byte tape. **The installed
Tapenade toolkit is never modified.** Upstream and patched file hashes are
recorded in the build manifest and in new experiment summaries. The toolkit
license is MIT. The tested installation is Tapenade 3.16 revision
`700ab029ecab52aff49be762419f13ef30d4ffe8`; its scheduler is unchanged.
`prepare_tapenade_runtime.py` checks the selected installation and patch on each
build, preserving timestamps of unchanged files to avoid recompilation. An
incompatible alignment patch causes the build to fail for review.

`TAPENADE_HOME` defaults to `$HOME/Install/tapenade_3.16`. For separate locations,
`TAPENADE` selects the generator executable and `ADKIT` selects the runtime
directory. Export `TAPENADE_HOME` or `ADKIT` when running Python so its automatic
Make invocation uses the same installation. Changing `TAPENADE` alone does not
change the runtime location. For example, override just the generator with:

```bash
make adjoint-generate TAPENADE=/path/to/tapenade
```

`make clean` preserves `tapenade_out/`, tests, documentation, and saved experiment
outputs. Raw generator output and staged input copies are disposable files under
`build/tapenade-generation/`.

## Project layout

C sources and headers remain at the project root. Only generated adjoint files
and their manifest are kept in `tapenade_out/`.

```text
swe2d_4dvar.c/.h          Checked C API and adjoint driver
swe_primal.c/.h           Forward kernels and scalar observation misfit
swe.py                   NumPy/ctypes interface and automatic library build
assimilate.py            Covariance, optimization, gradient checks, experiment
plot_4dvar_result.py      Figures and optional animation from saved results
Makefile                 Build and verification targets
generate_adjoint.py      Tapenade invocation and source manifest
fix_tapenade_c.py         Checked mechanical repairs to generated C
check_generated.py       Detect stale or modified generated sources
prepare_tapenade_runtime.py  Stage and patch the installed runtime
tapenade_alignment.patch Byte-tape alignment fixes
tapenade_out/
  swe_adjoint_b.c         Generated reverse model and binomial loop
  swe_primal_b.h          Generated declarations
  manifest.json          Generator information and source hashes
tests/
  reference/             Frozen manual adjoint used by numerical tests
  checkpoint_schedule.c  Standalone native scheduler verification
build/                  Disposable build and generation files
  tapenade_runtime/      Patched runtime, objects, toolkit license, provenance
  tapenade-generation/   Staged inputs, raw generator output, generation log
```

The unused `baseline/` directory and its comparison script have been removed.
Previously saved comparison figures and tables remain historical results.
`tests/reference/` is still required: it provides an independent gradient oracle.

## Tapenade generation and C repairs

`generate_adjoint.py` differentiates the scalar observation cost with respect to
the entire initial state, using the root `swe_misfit(cost)/(initial)`. The reverse
routine receives `costb=1` as its seed. It does not differentiate the Python
background covariance or optimizer.

The script stages the primal source and header in a temporary subdirectory
before invoking Tapenade. This preserves the C backend's header processing,
which otherwise omits the generated header when the input is in the working
directory. Copies of the exact inputs are saved in
`build/tapenade-generation/ad_primal/`.

The tested Tapenade 3.16 development C backend requires three mechanical repairs:

1. Remove four spurious `UnknownType` globals for variable-length array dimensions.
2. Pass `&action, &step` to `adBinomial_next`'s pointer arguments.
3. Save and restore all `3*n` elements of state and stage arrays at reverse turns,
   where the generator incorrectly emits scalar push/pop operations.

`fix_tapenade_c.py` checks every replacement count. It does not modify derivative
formulas or replace the scheduler. `generate_adjoint.py` checks for missing
output and Java exceptions, even if Tapenade exits with status zero. Unmodified
generator output is saved in `build/tapenade-generation/raw/`, alongside the
staged inputs and generation log. The source manifest stays in `tapenade_out/`
and records hashes of the primal, generation/repair scripts, and generated files.

## Model and objective

The model uses cell-centred conservative variables, Rusanov finite-volume
fluxes, gravity `g = 9.81`, and SSP-RK2 time stepping. Spatial accuracy is first
order and temporal accuracy is second order. Open boundaries copy the nearest
interior state. There is no bathymetry, forcing, or wetting/drying.

For the whitened initial-state control `x0 = xb + L z`,

```text
J(z) = 0.5 zᵀz + 0.5 Σ_k ||(H M_k(xb + L z) - y_k) / σobs||²
∇z J = z + Lᵀ λ0
```

`H` samples height. The C routine returns the observation cost and its gradient
`λ0` with respect to the full initial state. Python adds the background penalty
and applies `Lᵀ`. Tapenade differentiates the actual discrete fluxes, wave
speeds, boundary mappings, RK2 stages, and observation residuals. Flux max and
absolute-value operations are piecewise differentiable, so gradient validation
uses several perturbation sizes. Max-speed ties select the left branch; the
chosen derivative of absolute value at zero is zero.

`L` is a symmetric Gaussian filter in an orthonormal cosine basis, with separate
scales for `h`, `hu`, and `hv`. It supplies spatially correlated background errors
without forming a dense covariance matrix: `B = L Lᵀ`. The specified component
standard deviation is the square root of the spatially averaged point variance;
boundary variances may differ. `--length 0` gives diagonal covariance. There is
no cross-covariance between physical components or imposed dynamical balance.

Before invoking generated arithmetic, the C driver validates the full trajectory
using a constant number of state arrays. Every Euler/RK stage must be finite,
have depth above `1e-6`, and satisfy

```text
dt * [max(|u| + sqrt(g h))/dx + max(|v| + sqrt(g h))/dy] <= 0.9
```

Invalid optimization trials are rejected without clipping. Reduce `--dt` if a
finer grid violates this bound, adjusting `--stride` to preserve observation
times. Optimization failure saves diagnostics and exits with an error.

## C and Python interfaces

The public declarations are in [swe2d_4dvar.h](swe2d_4dvar.h). `swe_forward`
returns trajectories at observation times, including the initial state.
`swe_evaluate` returns the observation cost and initial-state gradient using
eight snapshot slots. To select a budget explicitly, use:

```c
int swe_evaluate_binomial(const double *initial,int nx,int ny,double lx,double ly,
    double dt,int nsteps,int stride,int nobs,const int *obs_idx,
    const double *observations,double obs_std,int checkpoint_budget,
    double *cost,double *gradient,double *diagnostics);
```

Set `gradient=NULL` for a cost-only evaluation. The budget must be 1..90 and
the total number of steps 1..1,000,000; a positive observation stride must divide
the step count. Diagnostics report minimum stage depth and maximum 2-D CFL.

The Python interface is `Model(..., checkpoint_budget=8)` with
`evaluate()` returning `(cost, gradient)`. The earlier statistics extension was
removed: `swe_evaluate_binomial` no longer takes a final statistics pointer,
`SweCheckpointStats` no longer exists, and `Model.last_checkpoint_stats` was
removed. C callers using that extension must rebuild and omit the old argument.
The `swe_forward` and `swe_evaluate` signatures are unchanged.

## Observation schedule and checkpoint budget

| Option | Default | Meaning |
|---|---:|---|
| `--dt` | 0.001 | Fixed model time step |
| `--stride` | 20 | Time steps between observations |
| `--cycles` | 15 | Observation times in one assimilation window |
| `--forecast-cycles` | 10 | Additional output times in the free forecast |
| `--checkpoint-budget` | 8 | Maximum simultaneous binomial snapshot slots, from 1 to 90 |

All observations in the assimilation window jointly update the initial state.
`--cycles` does not mean separate cycling analyses. Snapshot placement is chosen
by the binomial scheduler and need not coincide with observation times. Lower
budgets generally require more recomputation; the gradient should be unchanged.

The primal time loop carries the native directive:

```c
/* $AD BINOMIAL-CKP nt nsnap 0 */
for (it=0; it<nt; ++it) { ... }
```

The generated reverse loop calls the toolkit directly:

```c
adBinomial_init(nt, nsnap, 0);
while (adBinomial_next(&action, &step)) {
    switch (action) { /* generated save/restore/advance/reverse actions */ }
}
```

The tape and scheduler have global state, so a mutex in the C driver serializes
reverse evaluations, including concurrent ctypes calls. The driver checks that
the byte tape is empty after each reverse evaluation.

Each long-lived snapshot holds three full state arrays, an 8-byte cost, and a
4-byte loop index. For `N = nx*ny` cells and `c` live snapshots, its payload is
`c * (3 * 3*N*8 + 12)` bytes. Six state-sized work arrays, temporary reverse-turn
saves, and a local Euler/flux tape require additional memory. The runtime
allocates tape in 1 MiB blocks and can cache blocks between calls. Snapshot
payload is not total peak memory or process RSS.

For a 300-step trajectory, the native scheduler gives:

| Snapshot budget | Scheduled forward steps |
|---:|---:|
| 1 | 45150 |
| 2 | 4900 |
| 4 | 1938 |
| 8 | 1280 |
| 16 | 1029 |

These counts exclude the validation pass and the local Euler recomputation
inside each reverse RK2 step; they are not total floating-point operation counts.
Automatic differentiation is not claimed to be faster than the specialized
manual adjoint.

The production adjoint has no checkpoint-statistics wrapper. Scheduler bounds
and action counts, balanced saves/restores, and reverse-step counts are checked
separately in `tests/checkpoint_schedule.c`, which is not linked into `libswe.so`.
To compare gradients, timings, and standalone schedules across budgets:

```bash
python3 benchmark_checkpointing.py \
  --reference new_run --output checkpoint_benchmark
```

This reads a completed experiment, compares the generated gradient with the
manual oracle at its background and analysis states, and writes a CSV and
PNG/PDF tradeoff plot. Schedule counts come from an independent execution of
`adBinomial`; they are not counters inside the production adjoint.

## Outputs and validation

Each completed assimilation writes these files to the chosen output directory:

| File | Contents |
|---|---|
| `summary.json` | Configuration, convergence, errors, CFL/depth diagnostics, source hashes, and runtime provenance |
| `experiment.npz` | Truth/background/analysis trajectories, coordinates, sensors, observations, and final controls |
| `cost_history.csv` | Initial and accepted-iteration costs and gradient norms |
| `trajectory_metrics.csv` | Errors and mass at each saved physical time |
| `gradient_check.csv` | Signed directional finite differences and Taylor residuals |

State arrays have shape `(time, 3, ny, nx)` in component order `(h, hu, hv)`.
Plotting creates `var_4d_metrics`, `var_4d_snapshots`, `all_components_rmse`, and
`fd_adjoint_check` as PNG and PDF. `--movie` also creates `var_4d_result.gif`,
which animates physical time.

Tests cover all state components and corners, multiple observation times,
repeated sensors, checkpoint budgets, concurrent evaluations, conservation,
and the complete whitened objective. Native tests also exercise tapes spanning
multiple allocation blocks and compare all gradient components with the frozen
manual oracle at several budgets. Conservation is checked against the open
boundary flux; total mass need not remain constant.

The saved `var_output/` experiment uses the earlier manual adjoint;
`var_output_tapenade/` contains the earlier Tapenade run with scheduler counters.
That Tapenade run converged in 309 iterations by relative cost reduction,
reducing the objective from 11809.44 to 276.464216 and free-forecast depth RMSE
by 97.4%. Its final gradient infinity norm was `2.43e-5`. These archived results
are described in [RESULTS.md](RESULTS.md); new runs record their own source and
runtime provenance.

This is a same-model twin experiment with synthetic observations. Its forecast
improvement does not establish robustness to real model error or observational
bias. The covariance assumptions and limitations are discussed in
[adjoint.tex](adjoint.tex) and [adjoint.pdf](adjoint.pdf). Rebuild the document
with `bash buildpdf_4dvar.sh` using XeLaTeX, BibTeX, and the configured fonts.
