import glob
import os
import re

import numpy as np

os.makedirs("var_output/.mplconfig", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "var_output/.mplconfig")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DATA_DIR = "var_output"
TRUTH_PATTERN = os.path.join(DATA_DIR, "truth_cycle*.csv")
BACKGROUND_PATTERN = os.path.join(DATA_DIR, "background_cycle*.csv")
ANALYSIS_PATTERN = os.path.join(DATA_DIR, "analysis_cycle*.csv")
METRICS_FILE = os.path.join(DATA_DIR, "trajectory_metrics.csv")
COST_FILE = os.path.join(DATA_DIR, "cost_history.csv")
MISFIT_FILE = os.path.join(DATA_DIR, "iter_cycle_misfit.csv")
OBS_CONFIG_FILE = os.path.join(DATA_DIR, "obs_config.csv")

FIELD_NAME = "h"
MAKE_MOVIE = True
FPS = 3

OUTPUT_MOVIE_BASENAME = os.path.join(DATA_DIR, "var_4d_result")
OUTPUT_METRICS_FIG = os.path.join(DATA_DIR, "var_4d_metrics.png")
OUTPUT_SNAPSHOT_FIG = os.path.join(DATA_DIR, "var_4d_snapshots.png")
OUTPUT_MISFIT_FIG = os.path.join(DATA_DIR, "var_4d_cycle_misfit.png")

FIELD_CMAP = "YlGnBu"
ERROR_CMAP = "RdBu_r"
DIFF_CMAP = "coolwarm"


def extract_cycle_number(filename):
    match = re.search(r"cycle(\d+)\.csv$", filename)
    if not match:
        raise ValueError(f"Could not parse cycle number from {filename}")
    return int(match.group(1))


def read_field_csv(filename, field_name="h"):
    time_value = None
    with open(filename, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
        if first_line.startswith("# time="):
            time_value = float(first_line.split("=", 1)[1])

    data = np.genfromtxt(filename, delimiter=",", names=True, skip_header=1)
    if data.ndim == 0:
        data = np.array([data], dtype=data.dtype)

    i = data["i"].astype(int)
    j = data["j"].astype(int)
    nx = i.max() + 1
    ny = j.max() + 1

    x = data["x"].reshape(ny, nx)
    y = data["y"].reshape(ny, nx)
    z = data[field_name].reshape(ny, nx)
    return time_value, x, y, z


def load_optional_obs_config(filename):
    if not os.path.exists(filename):
        return None

    obs = np.genfromtxt(filename, delimiter=",", names=True)
    if obs.size == 0:
        return None
    if obs.ndim == 0:
        obs = np.array([obs], dtype=obs.dtype)
    return obs


def nice_limits(values, pad_frac=0.08, lower=None):
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 1.0)
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin), 1.0) * 0.1
        vmin -= delta
        vmax += delta
    pad = (vmax - vmin) * pad_frac
    vmin -= pad
    vmax += pad
    if lower is not None:
        vmin = max(vmin, lower)
    return vmin, vmax


def relative_improvement(background, analysis):
    return (background - analysis) / np.maximum(background, 1.0e-15)


truth_files = sorted(glob.glob(TRUTH_PATTERN), key=extract_cycle_number)
background_files = sorted(glob.glob(BACKGROUND_PATTERN), key=extract_cycle_number)
analysis_files = sorted(glob.glob(ANALYSIS_PATTERN), key=extract_cycle_number)

if not truth_files:
    raise RuntimeError(f"No files found for {TRUTH_PATTERN}")
if not background_files:
    raise RuntimeError(f"No files found for {BACKGROUND_PATTERN}")
if not analysis_files:
    raise RuntimeError(f"No files found for {ANALYSIS_PATTERN}")
if not (len(truth_files) == len(background_files) == len(analysis_files)):
    raise RuntimeError("Mismatch in number of truth/background/analysis files")
if not os.path.exists(METRICS_FILE):
    raise RuntimeError(f"Missing metrics file: {METRICS_FILE}")
if not os.path.exists(COST_FILE):
    raise RuntimeError(f"Missing cost history file: {COST_FILE}")
if not os.path.exists(MISFIT_FILE):
    raise RuntimeError(f"Missing cycle misfit file: {MISFIT_FILE}")

metrics = np.genfromtxt(METRICS_FILE, delimiter=",", names=True)
if metrics.ndim == 0:
    metrics = np.array([metrics], dtype=metrics.dtype)

cost_hist = np.genfromtxt(COST_FILE, delimiter=",", names=True)
if cost_hist.ndim == 0:
    cost_hist = np.array([cost_hist], dtype=cost_hist.dtype)

misfit_hist = np.genfromtxt(MISFIT_FILE, delimiter=",", names=True)
if misfit_hist.ndim == 0:
    misfit_hist = np.array([misfit_hist], dtype=misfit_hist.dtype)

obs_config = load_optional_obs_config(OBS_CONFIG_FILE)

cycles = []
times = []
truth_frames = []
background_frames = []
analysis_frames = []
background_err_frames = []
analysis_err_frames = []
improvement_frames = []

for tf, bf, af in zip(truth_files, background_files, analysis_files):
    cyc_t = extract_cycle_number(tf)
    cyc_b = extract_cycle_number(bf)
    cyc_a = extract_cycle_number(af)
    if len({cyc_t, cyc_b, cyc_a}) != 1:
        raise RuntimeError(f"Cycle mismatch: {tf}, {bf}, {af}")

    t_truth, x, y, z_truth = read_field_csv(tf, FIELD_NAME)
    t_back, _, _, z_back = read_field_csv(bf, FIELD_NAME)
    t_anal, _, _, z_anal = read_field_csv(af, FIELD_NAME)

    cycles.append(cyc_t)
    times.append(t_truth if t_truth is not None else np.nan)
    truth_frames.append(z_truth)
    background_frames.append(z_back)
    analysis_frames.append(z_anal)
    background_err_frames.append(z_back - z_truth)
    analysis_err_frames.append(z_anal - z_truth)
    improvement_frames.append(z_back - z_anal)

    if t_back is not None and not np.isclose(t_back, times[-1]):
        raise RuntimeError(f"Time mismatch between truth and background at cycle {cyc_t}")
    if t_anal is not None and not np.isclose(t_anal, times[-1]):
        raise RuntimeError(f"Time mismatch between truth and analysis at cycle {cyc_t}")

cycles = np.array(cycles)
times = np.array(times)

metric_cycles = np.atleast_1d(metrics["cycle"]).astype(int)
metric_times = np.atleast_1d(metrics["time"]).astype(float)
background_rmse = np.atleast_1d(metrics["background_rmse_h"]).astype(float)
analysis_rmse = np.atleast_1d(metrics["analysis_rmse_h"]).astype(float)
rmse_gain = background_rmse - analysis_rmse
rmse_ratio = analysis_rmse / np.maximum(background_rmse, 1.0e-15)

cost_iter = np.atleast_1d(cost_hist["iter"]).astype(int)
cost_vals = np.atleast_1d(cost_hist["cost"]).astype(float)
grad_norms = np.atleast_1d(cost_hist["grad_norm"]).astype(float)
step_lengths = np.atleast_1d(cost_hist["step_length"]).astype(float)
misfit_iters = np.unique(np.atleast_1d(misfit_hist["iter"]).astype(int))
misfit_cycles = np.unique(np.atleast_1d(misfit_hist["cycle"]).astype(int))

obs_cost_grid = np.full((len(misfit_cycles), len(misfit_iters)), np.nan)
obs_rms_grid = np.full((len(misfit_cycles), len(misfit_iters)), np.nan)
for row in np.atleast_1d(misfit_hist):
    i = np.where(misfit_cycles == int(row["cycle"]))[0][0]
    j = np.where(misfit_iters == int(row["iter"]))[0][0]
    obs_cost_grid[i, j] = float(row["obs_cost"])
    obs_rms_grid[i, j] = float(row["obs_rms"])

if not np.array_equal(metric_cycles, cycles):
    raise RuntimeError("Cycle mismatch between field files and trajectory_metrics.csv")

field_min = min(np.min(frame) for frame in truth_frames + background_frames + analysis_frames)
field_max = max(np.max(frame) for frame in truth_frames + background_frames + analysis_frames)
err_abs = max(
    np.max(np.abs(frame))
    for frame in (background_err_frames + analysis_err_frames)
)
err_abs = max(err_abs, 1.0e-12)
improvement_abs = max(np.max(np.abs(frame)) for frame in improvement_frames)
improvement_abs = max(improvement_abs, 1.0e-12)

x_extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]

obs_x = None
obs_y = None
if obs_config is not None:
    obs_i = np.atleast_1d(obs_config["i"]).astype(int)
    obs_j = np.atleast_1d(obs_config["j"]).astype(int)
    obs_x = x[obs_j, obs_i]
    obs_y = y[obs_j, obs_i]

plt.rcParams.update(
    {
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
        "axes.edgecolor": "#4a4a4a",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "grid.color": "#c9c2b6",
        "font.size": 10,
    }
)


def save_metrics_figure():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax_rmse = axes[0, 0]
    ax_ratio = axes[0, 1]
    ax_cost = axes[1, 0]
    ax_grad = axes[1, 1]

    ax_rmse.plot(metric_cycles, background_rmse, color="#b44b24", lw=2.4, label="Background")
    ax_rmse.plot(metric_cycles, analysis_rmse, color="#006d77", lw=2.6, label="4D-Var analysis")
    ax_rmse.fill_between(metric_cycles, analysis_rmse, background_rmse, where=(analysis_rmse <= background_rmse),
                         color="#8abf69", alpha=0.18)
    ax_rmse.set_title("Trajectory RMSE")
    ax_rmse.set_xlabel("Cycle")
    ax_rmse.set_ylabel(f"{FIELD_NAME} RMSE")
    ax_rmse.set_xlim(metric_cycles.min(), metric_cycles.max())
    ax_rmse.set_ylim(*nice_limits(np.r_[background_rmse, analysis_rmse], lower=0.0))
    ax_rmse.grid(True, alpha=0.5)
    ax_rmse.legend(frameon=False)

    ax_ratio.plot(metric_cycles, rmse_ratio, color="#5b3c88", lw=2.4, label="analysis / background")
    ax_ratio.axhline(1.0, color="#333333", lw=1.2, ls="--")
    ax_ratio.fill_between(metric_cycles, rmse_ratio, 1.0, where=(rmse_ratio <= 1.0),
                          color="#8abf69", alpha=0.22)
    ax_ratio.set_title("Relative Skill")
    ax_ratio.set_xlabel("Cycle")
    ax_ratio.set_ylabel("RMSE ratio")
    ax_ratio.set_xlim(metric_cycles.min(), metric_cycles.max())
    ax_ratio.set_ylim(*nice_limits(rmse_ratio, pad_frac=0.12))
    ax_ratio.grid(True, alpha=0.5)

    ax_cost.plot(cost_iter, cost_vals, color="#2a6f97", lw=2.5, marker="o", ms=5)
    ax_cost.set_title("Optimization Cost History")
    ax_cost.set_xlabel("Iteration")
    ax_cost.set_ylabel("Cost")
    ax_cost.set_xlim(cost_iter.min(), cost_iter.max())
    ax_cost.set_ylim(*nice_limits(cost_vals, lower=0.0))
    ax_cost.grid(True, alpha=0.5)

    ax_grad.plot(cost_iter, grad_norms, color="#7c2d12", lw=2.5, marker="o", ms=5, label="gradient norm")
    ax_grad_right = ax_grad.twinx()
    ax_grad_right.bar(cost_iter, step_lengths, width=0.42, color="#d8b365", alpha=0.28, label="step length")
    ax_grad.set_title("Gradient And Step Length")
    ax_grad.set_xlabel("Iteration")
    ax_grad.set_ylabel("gradient norm")
    ax_grad_right.set_ylabel("step length")
    ax_grad.set_xlim(cost_iter.min(), cost_iter.max())
    ax_grad.set_ylim(*nice_limits(grad_norms, lower=0.0))
    ax_grad_right.set_ylim(*nice_limits(step_lengths, lower=0.0))
    ax_grad.grid(True, alpha=0.5)

    handles = [ax_grad.lines[0], ax_grad_right.patches[0]]
    labels = ["gradient norm", "step length"]
    ax_grad.legend(handles, labels, frameon=False, loc="upper right")

    mean_ratio = float(np.mean(rmse_ratio[1:])) if len(rmse_ratio) > 1 else float(rmse_ratio[0])
    final_gain = float(rmse_gain[-1])
    best_cycle = int(metric_cycles[np.argmin(rmse_ratio)])

    fig.suptitle("4D-Var Numerical Validation", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        f"Mean analysis/background RMSE ratio over the window = {mean_ratio:.3f}; "
        f"largest relative gain occurs near cycle {best_cycle}; "
        f"final-cycle RMSE improvement = {final_gain:.3e}.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(OUTPUT_METRICS_FIG, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_snapshot_figure():
    snapshot_ids = [1, len(cycles) // 2, len(cycles) - 1]
    fig, axes = plt.subplots(4, len(snapshot_ids), figsize=(14, 10), constrained_layout=True)

    for col, frame in enumerate(snapshot_ids):
        cyc = cycles[frame]
        t = metric_times[frame]

        im0 = axes[0, col].imshow(
            truth_frames[frame], origin="lower", cmap=FIELD_CMAP,
            vmin=field_min, vmax=field_max, extent=x_extent,
            interpolation="nearest", aspect="auto"
        )
        im1 = axes[1, col].imshow(
            background_err_frames[frame], origin="lower", cmap=ERROR_CMAP,
            vmin=-err_abs, vmax=err_abs, extent=x_extent,
            interpolation="nearest", aspect="auto"
        )
        im2 = axes[2, col].imshow(
            analysis_err_frames[frame], origin="lower", cmap=ERROR_CMAP,
            vmin=-err_abs, vmax=err_abs, extent=x_extent,
            interpolation="nearest", aspect="auto"
        )
        im3 = axes[3, col].imshow(
            improvement_frames[frame], origin="lower", cmap=DIFF_CMAP,
            vmin=-improvement_abs, vmax=improvement_abs, extent=x_extent,
            interpolation="nearest", aspect="auto"
        )

        axes[0, col].set_title(f"cycle={cyc}, t={t:.2f}\nTruth")
        axes[1, col].set_title(f"Background error\nRMSE={background_rmse[frame]:.3e}")
        axes[2, col].set_title(f"4D-Var error\nRMSE={analysis_rmse[frame]:.3e}")
        axes[3, col].set_title(f"Background - 4D-Var\nmean gain={np.mean(improvement_frames[frame]):.2e}")

        for row in range(4):
            axes[row, col].set_xlabel("x")
            axes[row, col].set_ylabel("y")
            if obs_x is not None:
                axes[row, col].scatter(obs_x, obs_y, s=18, facecolors="none",
                                       edgecolors="#111111", linewidths=0.6)

    cbar0 = fig.colorbar(im0, ax=axes[0, :], fraction=0.02, pad=0.02)
    cbar0.set_label(FIELD_NAME)
    cbar1 = fig.colorbar(im1, ax=axes[1:3, :], fraction=0.02, pad=0.02)
    cbar1.set_label("state - truth")
    cbar2 = fig.colorbar(im3, ax=axes[3, :], fraction=0.02, pad=0.02)
    cbar2.set_label("background - analysis")

    fig.suptitle("Representative 4D-Var Window Snapshots", fontsize=15, fontweight="bold")
    fig.savefig(OUTPUT_SNAPSHOT_FIG, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_cycle_misfit_figure():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    ax_heat = axes[0]
    ax_lines = axes[1]

    im = ax_heat.imshow(
        obs_rms_grid,
        origin="lower",
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
        extent=[
            float(misfit_iters.min()) - 0.5,
            float(misfit_iters.max()) + 0.5,
            float(misfit_cycles.min()) - 0.5,
            float(misfit_cycles.max()) + 0.5,
        ],
    )
    ax_heat.set_title("Per-Cycle Observation Misfit")
    ax_heat.set_xlabel("Optimization iteration")
    ax_heat.set_ylabel("Assimilation cycle")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02).set_label("obs RMS")

    sample_cycles = sorted({1, len(misfit_cycles) // 2, len(misfit_cycles)})
    palette = ["#b44b24", "#006d77", "#5b3c88"]
    for color, cyc in zip(palette, sample_cycles):
        idx = np.where(misfit_cycles == cyc)[0][0]
        ax_lines.plot(misfit_iters, obs_rms_grid[idx], "-o", lw=2.0, ms=4, color=color, label=f"cycle {cyc}")

    mean_obs_rms = np.nanmean(obs_rms_grid, axis=0)
    ax_lines.plot(misfit_iters, mean_obs_rms, "-s", lw=2.5, ms=4, color="#222222", label="mean over cycles")
    ax_lines.set_title("Selected Cycle Misfit Traces")
    ax_lines.set_xlabel("Optimization iteration")
    ax_lines.set_ylabel("obs RMS")
    ax_lines.set_xlim(misfit_iters.min(), misfit_iters.max())
    ax_lines.set_ylim(*nice_limits(obs_rms_grid[np.isfinite(obs_rms_grid)], lower=0.0))
    ax_lines.grid(True, alpha=0.5)
    ax_lines.legend(frameon=False)

    fig.suptitle("4D-Var Iteration-By-Iteration Observation Misfit", fontsize=15, fontweight="bold")
    fig.savefig(OUTPUT_MISFIT_FIG, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)


save_metrics_figure()
save_snapshot_figure()
save_cycle_misfit_figure()

fig = plt.figure(figsize=(18, 10), constrained_layout=True)
mosaic = [
    ["truth", "bg_err", "an_err", "improve"],
    ["rmse", "rmse", "cost", "grad"],
]
axes = fig.subplot_mosaic(mosaic)

ax_truth = axes["truth"]
ax_bg_err = axes["bg_err"]
ax_an_err = axes["an_err"]
ax_improve = axes["improve"]
ax_rmse = axes["rmse"]
ax_cost = axes["cost"]
ax_grad = axes["grad"]

im_truth = ax_truth.imshow(
    truth_frames[0], origin="lower", cmap=FIELD_CMAP,
    vmin=field_min, vmax=field_max, extent=x_extent,
    interpolation="nearest", aspect="auto"
)
im_bg_err = ax_bg_err.imshow(
    background_err_frames[0], origin="lower", cmap=ERROR_CMAP,
    vmin=-err_abs, vmax=err_abs, extent=x_extent,
    interpolation="nearest", aspect="auto"
)
im_an_err = ax_an_err.imshow(
    analysis_err_frames[0], origin="lower", cmap=ERROR_CMAP,
    vmin=-err_abs, vmax=err_abs, extent=x_extent,
    interpolation="nearest", aspect="auto"
)
im_improve = ax_improve.imshow(
    improvement_frames[0], origin="lower", cmap=DIFF_CMAP,
    vmin=-improvement_abs, vmax=improvement_abs, extent=x_extent,
    interpolation="nearest", aspect="auto"
)

for ax in (ax_truth, ax_bg_err, ax_an_err, ax_improve):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if obs_x is not None:
        ax.scatter(obs_x, obs_y, s=22, facecolors="none", edgecolors="#111111", linewidths=0.8)

ax_truth.set_title("Truth Field")
ax_bg_err.set_title("Background Error")
ax_an_err.set_title("4D-Var Error")
ax_improve.set_title("Background - 4D-Var")

fig.colorbar(im_truth, ax=ax_truth, fraction=0.046, pad=0.02).set_label(FIELD_NAME)
fig.colorbar(im_bg_err, ax=[ax_bg_err, ax_an_err], fraction=0.046, pad=0.02).set_label("state - truth")
fig.colorbar(im_improve, ax=ax_improve, fraction=0.046, pad=0.02).set_label("background - analysis")

(line_bg_rmse,) = ax_rmse.plot(metric_cycles[:1], background_rmse[:1], color="#b44b24", lw=2.4, label="Background")
(line_an_rmse,) = ax_rmse.plot(metric_cycles[:1], analysis_rmse[:1], color="#006d77", lw=2.6, label="4D-Var")
(marker_bg_rmse,) = ax_rmse.plot([metric_cycles[0]], [background_rmse[0]], "o", ms=7, color="#b44b24")
(marker_an_rmse,) = ax_rmse.plot([metric_cycles[0]], [analysis_rmse[0]], "o", ms=7, color="#006d77")
rmse_cursor = ax_rmse.axvline(metric_cycles[0], color="#222222", lw=1.0, alpha=0.5)
ax_rmse.set_title("Window RMSE History")
ax_rmse.set_xlabel("Cycle")
ax_rmse.set_ylabel(f"{FIELD_NAME} RMSE")
ax_rmse.set_xlim(metric_cycles.min(), metric_cycles.max())
ax_rmse.set_ylim(*nice_limits(np.r_[background_rmse, analysis_rmse], lower=0.0))
ax_rmse.grid(True, alpha=0.5)
ax_rmse.legend(frameon=False, loc="upper right")

(line_cost,) = ax_cost.plot(cost_iter, cost_vals, color="#2a6f97", lw=2.5, marker="o", ms=5)
(cost_marker,) = ax_cost.plot([cost_iter[0]], [cost_vals[0]], "o", ms=8, color="#2a6f97")
ax_cost.set_title("Optimization Cost")
ax_cost.set_xlabel("Iteration")
ax_cost.set_ylabel("Cost")
ax_cost.set_xlim(cost_iter.min(), cost_iter.max())
ax_cost.set_ylim(*nice_limits(cost_vals, lower=0.0))
ax_cost.grid(True, alpha=0.5)

(line_grad,) = ax_grad.plot(cost_iter, grad_norms, color="#7c2d12", lw=2.5, marker="o", ms=5, label="gradient norm")
ax_grad_right = ax_grad.twinx()
bars_step = ax_grad_right.bar(cost_iter, np.zeros_like(step_lengths), width=0.42, color="#d8b365", alpha=0.28)
ax_grad.set_title("Gradient And Step Length")
ax_grad.set_xlabel("Iteration")
ax_grad.set_ylabel("gradient norm")
ax_grad_right.set_ylabel("step length")
ax_grad.set_xlim(cost_iter.min(), cost_iter.max())
ax_grad.set_ylim(*nice_limits(grad_norms, lower=0.0))
ax_grad_right.set_ylim(*nice_limits(step_lengths, lower=0.0))
ax_grad.grid(True, alpha=0.5)
ax_grad.legend([line_grad, bars_step[0]], ["gradient norm", "step length"], frameon=False, loc="upper right")

panel_note = ax_improve.text(
    0.02,
    0.98,
    "",
    transform=ax_improve.transAxes,
    va="top",
    ha="left",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=(1, 1, 1, 0.82), edgecolor="#c5b8a8", alpha=0.95),
)
suptitle = fig.suptitle("", fontsize=15, fontweight="bold", y=1.02)


def update(frame):
    im_truth.set_data(truth_frames[frame])
    im_bg_err.set_data(background_err_frames[frame])
    im_an_err.set_data(analysis_err_frames[frame])
    im_improve.set_data(improvement_frames[frame])

    line_bg_rmse.set_data(metric_cycles[: frame + 1], background_rmse[: frame + 1])
    line_an_rmse.set_data(metric_cycles[: frame + 1], analysis_rmse[: frame + 1])
    marker_bg_rmse.set_data([metric_cycles[frame]], [background_rmse[frame]])
    marker_an_rmse.set_data([metric_cycles[frame]], [analysis_rmse[frame]])
    rmse_cursor.set_xdata([metric_cycles[frame], metric_cycles[frame]])

    cost_marker.set_data([cost_iter[min(frame, len(cost_iter) - 1)]], [cost_vals[min(frame, len(cost_vals) - 1)]])
    for idx, bar in enumerate(bars_step):
        bar.set_height(step_lengths[idx] if idx <= min(frame, len(step_lengths) - 1) else 0.0)

    gain = rmse_gain[frame]
    rel_gain = relative_improvement(background_rmse[frame], analysis_rmse[frame])
    panel_note.set_text(
        f"Current cycle\n"
        f"Background RMSE: {background_rmse[frame]:.3e}\n"
        f"4D-Var RMSE:     {analysis_rmse[frame]:.3e}\n"
        f"Absolute gain:   {gain:.3e}\n"
        f"Relative gain:   {rel_gain:.2%}"
    )

    suptitle.set_text(
        f"2D SWE 4D-Var Window Reconstruction | {FIELD_NAME} | "
        f"cycle={cycles[frame]} | time={metric_times[frame]:.5f}"
    )

    return (
        im_truth,
        im_bg_err,
        im_an_err,
        im_improve,
        line_bg_rmse,
        line_an_rmse,
        marker_bg_rmse,
        marker_an_rmse,
        rmse_cursor,
        line_cost,
        cost_marker,
        line_grad,
        panel_note,
        suptitle,
    )


if MAKE_MOVIE:
    animation = FuncAnimation(fig, update, frames=len(cycles), interval=1000 / FPS, blit=False)
    try:
        animation.save(f"{OUTPUT_MOVIE_BASENAME}.mp4", writer="ffmpeg", fps=FPS)
        print(f"Saved {OUTPUT_MOVIE_BASENAME}.mp4")
    except Exception as exc:
        print("MP4 save failed:", exc)
        print("Trying GIF instead...")
        animation.save(f"{OUTPUT_MOVIE_BASENAME}.gif", writer="pillow", fps=FPS)
        print(f"Saved {OUTPUT_MOVIE_BASENAME}.gif")
else:
    for frame in range(len(cycles)):
        update(frame)
        plt.pause(0.6)
    plt.show()
