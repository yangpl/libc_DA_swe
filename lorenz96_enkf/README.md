# lorenz96_enkf

EnKF vs LETKF twin experiment for the Lorenz-96 model, ported from the
`swe2d_enkf` framework (perturbed-observation EnKF vs deterministic
square-root LETKF comparison).

**Naming** (simplified): "EnKF" here denotes the stochastic
perturbed-observation EnKF (Burgers 1998); "LETKF" denotes the
deterministic square-root filter applied locally (Whitaker & Hamill 2002;
Hunt et al. 2007), with Gaspari–Cohn covariance localization — the "L"
is now genuinely active.

## Model

$$
\frac{dx_k}{dt} = x_{k-1}(x_{k+1} - x_{k-2}) - x_k + F,
\qquad k = 1,\dots,N,
\qquad x_{k+N}=x_k
$$

Integrated with 4th-order Runge–Kutta (dt = 0.05; 1 time unit ≈ 5 days).

## Experiment

| Parameter | Value | Notes |
|---|---|---|
| N (state dimension) | 40 | Standard Lorenz-96 |
| F (forcing) | 8.0 | Chaotic regime |
| dt | 0.05 | RK4 time step (≈ 6 hours) |
| Spin-up | 1000 steps | 50 time units |
| Ne (ensemble size) | 40 | Localization makes this modest ensemble viable (see below) |
| Observations | every 4th variable (10) | |
| obs error std | 1.0 | |
| Assimilation window | 0.05 | 1 RK4 step |
| Cycles | 200 | 10 time units total |
| **LOC_RADIUS** | 3.0 | Gaspari–Cohn localization half-width (grid points); 0.0 = no localization |
| INFLATION | 1.0 | Multiplicative prior inflation; 1.0 = none |

### Analysis

Both filters use **Gaspari–Cohn (1999) covariance localization**
(half-width `LOC_RADIUS` = 3 grid points, cyclic distance), which tapers
spurious long-range sample correlations that dominate small-ensemble
covariances.

**EnKF** (stochastic, perturbed observations; Burgers 1998): each member
is updated with a randomly perturbed observation vector and the *localized*
gain (Houtekamer & Mitchell 2001)

$$\mathbf{K} = (\boldsymbol{\rho} \circ \mathbf{P}\mathbf{H}^\mathsf{T})
[\mathbf{H}(\boldsymbol{\rho} \circ \mathbf{P}\mathbf{H}^\mathsf{T}) + \mathbf{R}]^{-1},$$

with $\boldsymbol{\rho}$ the Gaspari–Cohn taper of each state–observation pair.

**LETKF** (deterministic square-root; Whitaker & Hamill 2002; Hunt et al.
2007): the ensemble-subspace square-root transform is recomputed for each
state variable $i$ individually, using only the observations weighted by
the Gaspari–Cohn taper $w_p(i) = \mathrm{GC}(\mathrm{dist}(i, \mathrm{obs}_p))$
(R-localization).  No observation perturbations are needed, so sampling
error from perturbed observations is avoided.

> **Why localization?**  For the classic Lorenz-96 setup (N=40, observations
> every 4th variable, σ_o = 1, 6-hourly updates) an ensemble of ~20–40
> members is too small to estimate background covariances reliably:
> without localization, spurious long-range sample correlations dominate
> the gain and both filters diverge after a few dozen cycles.  The
> Gaspari–Cohn taper (radius 3) smoothly zeroes correlations beyond its
> compact support (~11 grid points) and restores stability.  At Ne=40 with
> `LOC_RADIUS=3` both filters are well-tuned: steady-state
> spread/RMSE ≈ 0.92 (EnKF) and 1.04 (LETKF), with the LETKF's analysis
> RMSE ≈ 0.53 vs the EnKF's ≈ 0.58.  Set `NE=20` to see the filters
> struggle, or `LOC_RADIUS=0` to disable localization and reproduce the
> divergence.
>
> Set `INFLATION > 1.0` to add multiplicative prior covariance inflation
> (usually unnecessary once localization is applied; too much inflation
> degrades both filters).

The LETKF here uses the ensemble-subspace formulation applied **locally**:
for each state variable $i$, the transform is recomputed with observation
weights $w_p(i) = \mathrm{GC}(\mathrm{dist}(i, \mathrm{obs}_p))$:
\begin{align}
\mathbf{T}_e(i) &= \mathbf{I} + \sum_p \frac{w_p(i)}{\mathtt{Ne}-1}
  \mathbf{Y}'_p \mathbf{Y}'_p^{\mathsf{T}} \mathbf{R}^{-1} \\
\bar{x}^a(i) &= \bar{x}^f(i)
  + \frac{1}{\sqrt{\mathtt{Ne}-1}}
  \mathbf{A}(i,:)\, \mathbf{T}_e(i)^{-1}
  \sum_p \frac{w_p(i)}{\mathtt{Ne}-1}
  \mathbf{Y}'_p^\mathsf{T} \mathbf{R}^{-1}
  (y_p - \bar{y}^f_p) \\
\mathbf{A}^a(i,:) &= \mathbf{A}(i,:)\, \mathbf{T}_e(i)^{-1/2} \qquad
  (\text{eigen-decomposition } \mathbf{T}_e = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^\mathsf{T}) \\
\mathbf{x}^{(e),a}(i) &= \bar{x}^a(i) + \mathbf{A}^a(i,:)_e
\end{align}
where $\mathbf{A}$ ($n\times\mathtt{Ne}$) contains the raw anomalies,
$\mathbf{Y}' = \mathbf{H}\mathbf{A}$ the observed anomalies, and the
Gaspari–Cohn weights make observations beyond the localization radius
irrelevant for each state variable.

## Source layout

The code is split into functional modules, each with its own header:

| File | Contents |
|---|---|
| `main.c` | experiment driver: truth spin-up, cycling loop, metrics output |
| `rng.c` / `rng.h` | pseudo-random number generation (LCG + Box–Muller) |
| `l96.c` / `l96.h` | Lorenz-96 forward model and RK4 time integration |
| `linalg.c` / `linalg.h` | Gauss–Jordan inverse, Jacobi eigensolver, Gaspari–Cohn taper |
| `filter.c` / `filter.h` | **EnKF** (stochastic, localized gain) and **LETKF** (local square-root) analyses |
| `diag.c` / `diag.h` | diagnostics (RMSE, spread, innovations) and CSV output helpers |
| `plot_result.py` | diagnostic figures |

The Makefile compiles each module to an object file and links them into the
`lorenz96_enkf` executable.

## Build and run

```bash
make
./lorenz96_enkf
python3 plot_result.py
```

Or use the convenience script:

```bash
bash run.sh
```

## Output

| File | Contents |
|---|---|
| `output/metrics.csv` | per-cycle RMSE, spread, innovation RMS for both filters |
| `output/obs_config.csv` | observation index map |
| `output/state_history.csv` | full state-vector history (truth + ensemble means) |
| `lorenz96_enkf_filter.png` | filter diagnostics: RMSE, spread, spread/RMSE consistency, innovations |
| `lorenz96_enkf_state.png` | state diagnostics: Hovmöller of truth and analysis errors, final-cycle snapshot |