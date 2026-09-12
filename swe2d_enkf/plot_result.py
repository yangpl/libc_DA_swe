import glob
import os
import re

import numpy as np

os.makedirs("comparison_output/.mplconfig", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "comparison_output/.mplconfig")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

DATA_DIR = "comparison_output"
TRUTH_PATTERN = os.path.join(DATA_DIR, "truth_cycle*.csv")
ENKF_PATTERN = os.path.join(DATA_DIR, "stoch_enkf_mean_cycle*.csv")
ENSRF_PATTERN = os.path.join(DATA_DIR, "ensrf_mean_cycle*.csv")
METRICS_FILE = os.path.join(DATA_DIR, "metrics.csv")
OBS_CONFIG_FILE = os.path.join(DATA_DIR, "obs_config.csv")

MAKE_MOVIE = True
FPS = 3
FIELD_NAME = "h"
OUTPUT_MOVIE_BASENAME = os.path.join(DATA_DIR, "comparison_result")

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


truth_files = sorted(glob.glob(TRUTH_PATTERN), key=extract_cycle_number)
enkf_files = sorted(glob.glob(ENKF_PATTERN), key=extract_cycle_number)
ensrf_files = sorted(glob.glob(ENSRF_PATTERN), key=extract_cycle_number)

if not truth_files:
    raise RuntimeError(f"No files found for {TRUTH_PATTERN}")
if not enkf_files:
    raise RuntimeError(f"No files found for {ENKF_PATTERN}")
if not ensrf_files:
    raise RuntimeError(f"No files found for {ENSRF_PATTERN}")
if not (len(truth_files) == len(enkf_files) == len(ensrf_files)):
    raise RuntimeError("Mismatch in number of truth/EnKF/EnSRF files")
if not os.path.exists(METRICS_FILE):
    raise RuntimeError(f"Missing metrics file: {METRICS_FILE}")

metrics = np.genfromtxt(METRICS_FILE, delimiter=",", names=True)
if metrics.ndim == 0:
    metrics = np.array([metrics], dtype=metrics.dtype)

obs_config = load_optional_obs_config(OBS_CONFIG_FILE)

cycles = []
times = []
truth_frames = []
enkf_frames = []
ensrf_frames = []
enkf_err_frames = []
ensrf_err_frames = []
method_diff_frames = []

for tf, ef, sf in zip(truth_files, enkf_files, ensrf_files):
    cyc_t = extract_cycle_number(tf)
    cyc_e = extract_cycle_number(ef)
    cyc_s = extract_cycle_number(sf)
    if len({cyc_t, cyc_e, cyc_s}) != 1:
        raise RuntimeError(f"Cycle mismatch: {tf}, {ef}, {sf}")

    t_truth, x, y, z_truth = read_field_csv(tf, FIELD_NAME)
    t_enkf, _, _, z_enkf = read_field_csv(ef, FIELD_NAME)
    t_ensrf, _, _, z_ensrf = read_field_csv(sf, FIELD_NAME)

    cycles.append(cyc_t)
    times.append(t_truth if t_truth is not None else np.nan)
    truth_frames.append(z_truth)
    enkf_frames.append(z_enkf)
    ensrf_frames.append(z_ensrf)
    enkf_err_frames.append(z_enkf - z_truth)
    ensrf_err_frames.append(z_ensrf - z_truth)
    method_diff_frames.append(z_ensrf - z_enkf)

    if t_enkf is not None and not np.isclose(t_enkf, times[-1]):
        raise RuntimeError(f"Time mismatch between truth and EnKF at cycle {cyc_t}")
    if t_ensrf is not None and not np.isclose(t_ensrf, times[-1]):
        raise RuntimeError(f"Time mismatch between truth and EnSRF at cycle {cyc_t}")

cycles = np.array(cycles)
times = np.array(times)

metric_cycles = np.atleast_1d(metrics["cycle"]).astype(int)
metric_times = np.atleast_1d(metrics["time"]).astype(float)
if not np.array_equal(metric_cycles, cycles):
    raise RuntimeError("Cycle mismatch between field files and metrics.csv")

enkf_rmse_f = np.atleast_1d(metrics["enkf_rmse_f"]).astype(float)
enkf_rmse_a = np.atleast_1d(metrics["enkf_rmse_a"]).astype(float)
ensrf_rmse_f = np.atleast_1d(metrics["ensrf_rmse_f"]).astype(float)
ensrf_rmse_a = np.atleast_1d(metrics["ensrf_rmse_a"]).astype(float)
enkf_spread_f = np.atleast_1d(metrics["enkf_spread_f"]).astype(float)
enkf_spread_a = np.atleast_1d(metrics["enkf_spread_a"]).astype(float)
ensrf_spread_f = np.atleast_1d(metrics["ensrf_spread_f"]).astype(float)
ensrf_spread_a = np.atleast_1d(metrics["ensrf_spread_a"]).astype(float)
enkf_innov = np.atleast_1d(metrics["enkf_innov_rms_f"]).astype(float)
ensrf_innov = np.atleast_1d(metrics["ensrf_innov_rms_f"]).astype(float)
enkf_clipped = np.atleast_1d(metrics["enkf_clipped"]).astype(float)
ensrf_clipped = np.atleast_1d(metrics["ensrf_clipped"]).astype(float)

field_min = min(np.min(frame) for frame in truth_frames + enkf_frames + ensrf_frames)
field_max = max(np.max(frame) for frame in truth_frames + enkf_frames + ensrf_frames)

err_abs = max(
    np.max(np.abs(frame))
    for frame in (enkf_err_frames + ensrf_err_frames)
)
err_abs = max(err_abs, 1.0e-12)

method_diff_abs = max(np.max(np.abs(frame)) for frame in method_diff_frames)
method_diff_abs = max(method_diff_abs, 1.0e-12)

x_extent = [float(np.min(x)), float(np.max(x)), float(np.min(y)), float(np.max(y))]

plt.rcParams.update(
    {
        "figure.facecolor": "#f6f3ea",
        "axes.facecolor": "#fbfaf6",
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

fig = plt.figure(figsize=(18, 10), constrained_layout=True)
mosaic = [
    ["truth", "enkf_err", "ensrf_err", "method_diff"],
    ["history", "history", "spread", "skill"],
]
axes = fig.subplot_mosaic(mosaic)

ax_truth = axes["truth"]
ax_enkf_err = axes["enkf_err"]
ax_ensrf_err = axes["ensrf_err"]
ax_method_diff = axes["method_diff"]
ax_history = axes["history"]
ax_spread = axes["spread"]
ax_skill = axes["skill"]

im_truth = ax_truth.imshow(
    truth_frames[0],
    origin="lower",
    cmap=FIELD_CMAP,
    vmin=field_min,
    vmax=field_max,
    extent=x_extent,
    interpolation="nearest",
    aspect="auto",
)
im_enkf_err = ax_enkf_err.imshow(
    enkf_err_frames[0],
    origin="lower",
    cmap=ERROR_CMAP,
    vmin=-err_abs,
    vmax=err_abs,
    extent=x_extent,
    interpolation="nearest",
    aspect="auto",
)
im_ensrf_err = ax_ensrf_err.imshow(
    ensrf_err_frames[0],
    origin="lower",
    cmap=ERROR_CMAP,
    vmin=-err_abs,
    vmax=err_abs,
    extent=x_extent,
    interpolation="nearest",
    aspect="auto",
)
im_method_diff = ax_method_diff.imshow(
    method_diff_frames[0],
    origin="lower",
    cmap=DIFF_CMAP,
    vmin=-method_diff_abs,
    vmax=method_diff_abs,
    extent=x_extent,
    interpolation="nearest",
    aspect="auto",
)

for ax in (ax_truth, ax_enkf_err, ax_ensrf_err, ax_method_diff):
    ax.set_xlabel("x")
    ax.set_ylabel("y")

ax_truth.set_title("Truth Field")
ax_enkf_err.set_title("Stochastic EnKF Error")
ax_ensrf_err.set_title("EnSRF Error")
ax_method_diff.set_title("EnSRF - EnKF")

if obs_config is not None:
    obs_i = np.atleast_1d(obs_config["i"]).astype(int)
    obs_j = np.atleast_1d(obs_config["j"]).astype(int)
    obs_x = x[obs_j, obs_i]
    obs_y = y[obs_j, obs_i]
    scatter_style = dict(s=22, facecolors="none", edgecolors="#111111", linewidths=0.8)
    for ax in (ax_truth, ax_enkf_err, ax_ensrf_err, ax_method_diff):
        ax.scatter(obs_x, obs_y, **scatter_style)

cbar_truth = fig.colorbar(im_truth, ax=ax_truth, fraction=0.046, pad=0.02)
cbar_truth.set_label(FIELD_NAME)
cbar_err = fig.colorbar(im_enkf_err, ax=[ax_enkf_err, ax_ensrf_err], fraction=0.046, pad=0.02)
cbar_err.set_label("analysis - truth")
cbar_diff = fig.colorbar(im_method_diff, ax=ax_method_diff, fraction=0.046, pad=0.02)
cbar_diff.set_label("EnSRF - EnKF")

history_colors = {
    "enkf": "#d55c00",
    "ensrf": "#007a78",
}

(line_enkf_rmse_f,) = ax_history.plot(
    metric_cycles[:1], enkf_rmse_f[:1], "--", lw=1.8, color=history_colors["enkf"], alpha=0.70, label="EnKF forecast RMSE"
)
(line_enkf_rmse_a,) = ax_history.plot(
    metric_cycles[:1], enkf_rmse_a[:1], "-", lw=2.6, color=history_colors["enkf"], label="EnKF analysis RMSE"
)
(line_ensrf_rmse_f,) = ax_history.plot(
    metric_cycles[:1], ensrf_rmse_f[:1], "--", lw=1.8, color=history_colors["ensrf"], alpha=0.70, label="EnSRF forecast RMSE"
)
(line_ensrf_rmse_a,) = ax_history.plot(
    metric_cycles[:1], ensrf_rmse_a[:1], "-", lw=2.6, color=history_colors["ensrf"], label="EnSRF analysis RMSE"
)
(marker_enkf_rmse,) = ax_history.plot(
    [metric_cycles[0]], [enkf_rmse_a[0]], "o", ms=7, color=history_colors["enkf"]
)
(marker_ensrf_rmse,) = ax_history.plot(
    [metric_cycles[0]], [ensrf_rmse_a[0]], "o", ms=7, color=history_colors["ensrf"]
)
history_cursor = ax_history.axvline(metric_cycles[0], color="#222222", lw=1.0, alpha=0.5)

ax_history.set_title("RMSE History")
ax_history.set_xlabel("DA cycle")
ax_history.set_ylabel(f"{FIELD_NAME} RMSE")
ax_history.set_xlim(metric_cycles.min(), metric_cycles.max())
ax_history.set_ylim(*nice_limits(np.r_[enkf_rmse_f, enkf_rmse_a, ensrf_rmse_f, ensrf_rmse_a], lower=0.0))
ax_history.grid(True, alpha=0.5)
ax_history.legend(loc="upper right", ncol=2, frameon=False)

(line_enkf_spread_f,) = ax_spread.plot(
    metric_cycles[:1], enkf_spread_f[:1], "--", lw=1.8, color=history_colors["enkf"], alpha=0.70, label="EnKF forecast spread"
)
(line_enkf_spread_a,) = ax_spread.plot(
    metric_cycles[:1], enkf_spread_a[:1], "-", lw=2.6, color=history_colors["enkf"], label="EnKF analysis spread"
)
(line_ensrf_spread_f,) = ax_spread.plot(
    metric_cycles[:1], ensrf_spread_f[:1], "--", lw=1.8, color=history_colors["ensrf"], alpha=0.70, label="EnSRF forecast spread"
)
(line_ensrf_spread_a,) = ax_spread.plot(
    metric_cycles[:1], ensrf_spread_a[:1], "-", lw=2.6, color=history_colors["ensrf"], label="EnSRF analysis spread"
)
spread_cursor = ax_spread.axvline(metric_cycles[0], color="#222222", lw=1.0, alpha=0.5)

ax_spread.set_title("Ensemble Spread")
ax_spread.set_xlabel("DA cycle")
ax_spread.set_ylabel(f"{FIELD_NAME} spread")
ax_spread.set_xlim(metric_cycles.min(), metric_cycles.max())
ax_spread.set_ylim(*nice_limits(np.r_[enkf_spread_f, enkf_spread_a, ensrf_spread_f, ensrf_spread_a], lower=0.0))
ax_spread.grid(True, alpha=0.5)
ax_spread.legend(loc="upper right", frameon=False, fontsize=8)

(line_enkf_innov,) = ax_skill.plot(
    metric_cycles[:1], enkf_innov[:1], "-", lw=2.2, color=history_colors["enkf"], label="EnKF innovation RMS"
)
(line_ensrf_innov,) = ax_skill.plot(
    metric_cycles[:1], ensrf_innov[:1], "-", lw=2.2, color=history_colors["ensrf"], label="EnSRF innovation RMS"
)
(marker_enkf_innov,) = ax_skill.plot(
    [metric_cycles[0]], [enkf_innov[0]], "o", ms=6, color=history_colors["enkf"]
)
(marker_ensrf_innov,) = ax_skill.plot(
    [metric_cycles[0]], [ensrf_innov[0]], "o", ms=6, color=history_colors["ensrf"]
)

ax_skill_right = ax_skill.twinx()
bar_width = 0.38
bars_enkf = ax_skill_right.bar(
    metric_cycles - bar_width / 2,
    np.zeros_like(enkf_clipped),
    width=bar_width,
    color=history_colors["enkf"],
    alpha=0.28,
    label="EnKF clipped",
)
bars_ensrf = ax_skill_right.bar(
    metric_cycles + bar_width / 2,
    np.zeros_like(ensrf_clipped),
    width=bar_width,
    color=history_colors["ensrf"],
    alpha=0.28,
    label="EnSRF clipped",
)
skill_cursor = ax_skill.axvline(metric_cycles[0], color="#222222", lw=1.0, alpha=0.5)

ax_skill.set_title("Innovation And Clipping")
ax_skill.set_xlabel("DA cycle")
ax_skill.set_ylabel("innovation RMS")
ax_skill.set_xlim(metric_cycles.min(), metric_cycles.max())
ax_skill.set_ylim(*nice_limits(np.r_[enkf_innov, ensrf_innov], lower=0.0))
ax_skill.grid(True, alpha=0.5)

clip_max = max(np.max(enkf_clipped), np.max(ensrf_clipped))
ax_skill_right.set_ylabel("clipped cells")
ax_skill_right.set_ylim(0.0, clip_max * 1.15 if clip_max > 0 else 1.0)

skill_handles = [line_enkf_innov, line_ensrf_innov, bars_enkf[0], bars_ensrf[0]]
skill_labels = ["EnKF innovation RMS", "EnSRF innovation RMS", "EnKF clipped", "EnSRF clipped"]
ax_skill.legend(skill_handles, skill_labels, loc="upper right", frameon=False, fontsize=8)

panel_note = ax_method_diff.text(
    0.02,
    0.98,
    "",
    transform=ax_method_diff.transAxes,
    va="top",
    ha="left",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#c5b8a8", alpha=0.9),
)
suptitle = fig.suptitle("", fontsize=15, fontweight="bold", y=1.02)


def update(frame):
    im_truth.set_data(truth_frames[frame])
    im_enkf_err.set_data(enkf_err_frames[frame])
    im_ensrf_err.set_data(ensrf_err_frames[frame])
    im_method_diff.set_data(method_diff_frames[frame])

    line_enkf_rmse_f.set_data(metric_cycles[: frame + 1], enkf_rmse_f[: frame + 1])
    line_enkf_rmse_a.set_data(metric_cycles[: frame + 1], enkf_rmse_a[: frame + 1])
    line_ensrf_rmse_f.set_data(metric_cycles[: frame + 1], ensrf_rmse_f[: frame + 1])
    line_ensrf_rmse_a.set_data(metric_cycles[: frame + 1], ensrf_rmse_a[: frame + 1])
    marker_enkf_rmse.set_data([metric_cycles[frame]], [enkf_rmse_a[frame]])
    marker_ensrf_rmse.set_data([metric_cycles[frame]], [ensrf_rmse_a[frame]])
    history_cursor.set_xdata([metric_cycles[frame], metric_cycles[frame]])

    line_enkf_spread_f.set_data(metric_cycles[: frame + 1], enkf_spread_f[: frame + 1])
    line_enkf_spread_a.set_data(metric_cycles[: frame + 1], enkf_spread_a[: frame + 1])
    line_ensrf_spread_f.set_data(metric_cycles[: frame + 1], ensrf_spread_f[: frame + 1])
    line_ensrf_spread_a.set_data(metric_cycles[: frame + 1], ensrf_spread_a[: frame + 1])
    spread_cursor.set_xdata([metric_cycles[frame], metric_cycles[frame]])

    line_enkf_innov.set_data(metric_cycles[: frame + 1], enkf_innov[: frame + 1])
    line_ensrf_innov.set_data(metric_cycles[: frame + 1], ensrf_innov[: frame + 1])
    marker_enkf_innov.set_data([metric_cycles[frame]], [enkf_innov[frame]])
    marker_ensrf_innov.set_data([metric_cycles[frame]], [ensrf_innov[frame]])
    skill_cursor.set_xdata([metric_cycles[frame], metric_cycles[frame]])

    for bar, height in zip(bars_enkf, enkf_clipped[: frame + 1]):
        bar.set_height(height)
        bar.set_alpha(0.28)
    for bar in bars_enkf[frame + 1 :]:
        bar.set_height(0.0)

    for bar, height in zip(bars_ensrf, ensrf_clipped[: frame + 1]):
        bar.set_height(height)
        bar.set_alpha(0.28)
    for bar in bars_ensrf[frame + 1 :]:
        bar.set_height(0.0)

    rmse_gap = enkf_rmse_a[frame] - ensrf_rmse_a[frame]
    diff_rms = np.sqrt(np.mean(method_diff_frames[frame] ** 2))
    panel_note.set_text(
        f"Current cycle\n"
        f"EnKF RMSE:  {enkf_rmse_a[frame]:.3e}\n"
        f"EnSRF RMSE: {ensrf_rmse_a[frame]:.3e}\n"
        f"RMSE gap (E-S): {rmse_gap:.3e}\n"
        f"RMS(EnSRF-EnKF): {diff_rms:.3e}"
    )

    suptitle.set_text(
        f"2D SWE Data Assimilation Comparison | {FIELD_NAME} | "
        f"cycle={cycles[frame]} | time={metric_times[frame]:.5f}"
    )

    return (
        im_truth,
        im_enkf_err,
        im_ensrf_err,
        im_method_diff,
        line_enkf_rmse_f,
        line_enkf_rmse_a,
        line_ensrf_rmse_f,
        line_ensrf_rmse_a,
        marker_enkf_rmse,
        marker_ensrf_rmse,
        history_cursor,
        line_enkf_spread_f,
        line_enkf_spread_a,
        line_ensrf_spread_f,
        line_ensrf_spread_a,
        spread_cursor,
        line_enkf_innov,
        line_ensrf_innov,
        marker_enkf_innov,
        marker_ensrf_innov,
        skill_cursor,
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
