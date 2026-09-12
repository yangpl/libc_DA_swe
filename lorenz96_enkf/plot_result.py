#!/usr/bin/env python3
"""
Diagnostics figures for the Lorenz-96 EnKF/LETKF twin experiment.

Reads output/metrics.csv and output/state_history.csv and produces two
figures:

  lorenz96_enkf_filter.png   — filter diagnostics (RMSE, spread,
                               spread/RMSE consistency, innovations)
  lorenz96_enkf_state.png    — state diagnostics (Hovmoeller of truth and
                               analysis errors, final-cycle snapshot with
                               observation locations)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "output"
FIG_FILTER = "lorenz96_enkf_filter.png"
FIG_STATE = "lorenz96_enkf_state.png"

metrics_path = os.path.join(OUT, "metrics.csv")
history_path = os.path.join(OUT, "state_history.csv")

if not (os.path.exists(metrics_path) and os.path.exists(history_path)):
    sys.exit("Missing output files; run './lorenz96_enkf' first.  "
             "Expected in 'output/'.")

metrics = np.loadtxt(metrics_path, delimiter=",", skiprows=1)
hist = np.loadtxt(history_path, delimiter=",", skiprows=1)

if len(metrics) == 0:
    sys.exit("metrics.csv is empty — no data to plot.")

# --- metrics columns ------------------------------------------------------
cycle  = metrics[:, 0].astype(int)
t      = metrics[:, 1]
enkf_f  = metrics[:, 2];  enkf_a  = metrics[:, 3]
letkf_f  = metrics[:, 4];  letkf_a  = metrics[:, 5]
enkf_sf = metrics[:, 6];  enkf_sa = metrics[:, 7]
letkf_sf = metrics[:, 8];  letkf_sa = metrics[:, 9]
enkf_in = metrics[:, 10]; letkf_in = metrics[:, 11]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
})

C_ENKF = "C0"
C_LETKF = "C3"

# =========================================================================
# Figure 1 — filter diagnostics
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# (a) RMSE
ax = axes[0, 0]
ax.plot(cycle, enkf_f, color=C_ENKF, ls="--", lw=1, label="EnKF forecast")
ax.plot(cycle, enkf_a, color=C_ENKF, lw=1.5, label="EnKF analysis")
ax.plot(cycle, letkf_f, color=C_LETKF, ls="--", lw=1, label="LETKF forecast")
ax.plot(cycle, letkf_a, color=C_LETKF, lw=1.5, label="LETKF analysis")
ax.set_xlabel("cycle")
ax.set_ylabel("RMSE")
ax.set_title("(a) RMSE vs truth")
ax.legend()
ax.grid(alpha=0.3)

# (b) spread
ax = axes[0, 1]
ax.plot(cycle, enkf_sf, color=C_ENKF, ls="--", lw=1, label="EnKF forecast")
ax.plot(cycle, enkf_sa, color=C_ENKF, lw=1.5, label="EnKF analysis")
ax.plot(cycle, letkf_sf, color=C_LETKF, ls="--", lw=1, label="LETKF forecast")
ax.plot(cycle, letkf_sa, color=C_LETKF, lw=1.5, label="LETKF analysis")
ax.set_xlabel("cycle")
ax.set_ylabel("ensemble spread")
ax.set_title("(b) Ensemble spread")
ax.legend()
ax.grid(alpha=0.3)

# (c) EnKF: RMSE vs spread
ax = axes[0, 2]
ax.plot(cycle, enkf_a, color=C_ENKF, lw=1.5, label="RMSE$_a$")
ax.plot(cycle, enkf_sa, color=C_ENKF, ls="--", lw=1.2, label="spread$_a$")
ax.set_xlabel("cycle")
ax.set_ylabel("")
ax.set_title("(c) EnKF: spread vs error")
ax.legend()
ax.grid(alpha=0.3)

# (d) LETKF: RMSE vs spread
ax = axes[1, 0]
ax.plot(cycle, letkf_a, color=C_LETKF, lw=1.5, label="RMSE$_a$")
ax.plot(cycle, letkf_sa, color=C_LETKF, ls="--", lw=1.2, label="spread$_a$")
ax.set_xlabel("cycle")
ax.set_ylabel("")
ax.set_title("(d) LETKF: spread vs error")
ax.legend()
ax.grid(alpha=0.3)

# (e) innovation
ax = axes[1, 1]
ax.plot(cycle[1:], enkf_in[1:], color=C_ENKF, lw=1, label="EnKF")
ax.plot(cycle[1:], letkf_in[1:], color=C_LETKF, lw=1, label="LETKF")
ax.axhline(1.0, color="k", ls=":", lw=1, label=r"$\sigma_o$")
ax.set_xlabel("cycle")
ax.set_ylabel("innovation RMS")
ax.set_title("(e) Observation innovations")
ax.legend()
ax.grid(alpha=0.3)

# (f) spread/RMSE consistency
ax = axes[1, 2]
with np.errstate(divide="ignore", invalid="ignore"):
    ratio_enkf = enkf_sa / enkf_a
    ratio_letkf = letkf_sa / letkf_a
ax.plot(cycle, ratio_enkf, color=C_ENKF, lw=1.2, label="EnKF")
ax.plot(cycle, ratio_letkf, color=C_LETKF, lw=1.2, label="LETKF")
ax.axhline(1.0, color="k", ls=":", lw=1, label="well-tuned")
ax.set_ylim(0, 2)
ax.set_xlabel("cycle")
ax.set_ylabel("spread$_a$ / RMSE$_a$")
ax.set_title("(f) Spread/RMSE consistency (analysis)")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Lorenz-96 twin experiment: EnKF vs LETKF (filter diagnostics)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG_FILTER, dpi=150)
plt.close(fig)
print(f"Figure saved: {FIG_FILTER}")

# =========================================================================
# Figure 2 — state diagnostics
# =========================================================================
nstate = int(hist[:, 1].max()) + 1
ncyc = int(hist[:, 0].max()) + 1
nobs = 10                      # from OBS_EVERY = 4, NX = 40
obs_indices = np.arange(0, 40, 4)

truth = np.full((ncyc, nstate), np.nan)
mean_enkf = np.full((ncyc, nstate), np.nan)
mean_letkf = np.full((ncyc, nstate), np.nan)
for row in hist:
    c, k = int(row[0]), int(row[1])
    truth[c, k] = row[2]
    mean_enkf[c, k] = row[3]
    mean_letkf[c, k] = row[4]

err_enkf = mean_enkf - truth
err_letkf = mean_letkf - truth

# common color scale for the two error panels
vmax_err = max(np.nanmax(np.abs(err_enkf)), np.nanmax(np.abs(err_letkf)))

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (a) Hovmoeller of truth
ax = axes[0, 0]
im = ax.pcolormesh(truth.T, cmap="RdBu_r", shading="auto")
cb = fig.colorbar(im, ax=ax)
cb.set_label(r"$x_k$")
ax.set_xlabel("cycle  (1 cycle $\\approx$ 6 h)")
ax.set_ylabel("state index $k$")
ax.set_title("(a) Truth  $x^t(t)$")

# (b) LETKF analysis error Hovmoeller
ax = axes[0, 1]
im = ax.pcolormesh(err_letkf.T, cmap="RdBu_r", shading="auto",
                   vmin=-vmax_err, vmax=vmax_err)
cb = fig.colorbar(im, ax=ax)
cb.set_label("error")
ax.set_xlabel("cycle  (1 cycle $\\approx$ 6 h)")
ax.set_ylabel("state index $k$")
ax.set_title("(b) LETKF analysis error  $\\bar{x}^a - x^t$")

# (c) EnKF analysis error Hovmoeller
ax = axes[1, 0]
im = ax.pcolormesh(err_enkf.T, cmap="RdBu_r", shading="auto",
                   vmin=-vmax_err, vmax=vmax_err)
cb = fig.colorbar(im, ax=ax)
cb.set_label("error")
ax.set_xlabel("cycle  (1 cycle $\\approx$ 6 h)")
ax.set_ylabel("state index $k$")
ax.set_title("(c) EnKF analysis error  $\\bar{x}^a - x^t$")

# (d) final-cycle snapshot with observation locations
ax = axes[1, 1]
k = np.arange(nstate)
ax.plot(k, truth[-1], "k-", lw=1.5, label="truth")
ax.plot(k, mean_enkf[-1], color=C_ENKF, lw=1.2, label="EnKF mean")
ax.plot(k, mean_letkf[-1], color=C_LETKF, lw=1.2, label="LETKF mean")
ax.scatter(obs_indices, truth[-1, obs_indices], s=28, facecolors="none",
           edgecolors="k", marker="o", zorder=3,
           label="observed locations")
ax.set_xlabel("state index $k$")
ax.set_ylabel(r"$x_k$")
ax.set_title("(d) Final-cycle analysis snapshot")
ax.legend()
ax.grid(alpha=0.3)

fig.suptitle("Lorenz-96 twin experiment: EnKF vs LETKF (state diagnostics)",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(FIG_STATE, dpi=150)
plt.close(fig)
print(f"Figure saved: {FIG_STATE}")

# =========================================================================
# Summary statistics
# =========================================================================
last = metrics[-1]
print("")
print(f"Final cycle {int(last[0]):>4d}:  "
      f"EnKF RMSE_a = {last[3]:.4f}   LETKF RMSE_a = {last[5]:.4f}")

if len(metrics) >= 50:
    ss = metrics[-50:]
    r_enkf = ss[:, 7].mean() / ss[:, 3].mean()
    r_letkf = ss[:, 9].mean() / ss[:, 5].mean()
    print(f"Steady-state mean (last 50 cycles):")
    print(f"  EnKF  RMSE_a = {ss[:, 3].mean():.4f}  spread_a = {ss[:, 7].mean():.4f}"
          f"  (spread/RMSE = {r_enkf:.2f})")
    print(f"  LETKF RMSE_a = {ss[:, 5].mean():.4f}  spread_a = {ss[:, 9].mean():.4f}"
          f"  (spread/RMSE = {r_letkf:.2f})")
else:
    print("(fewer than 50 cycles — steady-state mean not computed)")