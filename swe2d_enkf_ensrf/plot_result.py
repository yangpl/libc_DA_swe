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

MAKE_MOVIE = True
FPS = 3
FIELD_NAME = "h"
OUTPUT_MOVIE_BASENAME = os.path.join(DATA_DIR, "comparison_result")


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

cycles = []
times = []
truth_frames = []
enkf_frames = []
ensrf_frames = []
enkf_err_frames = []
ensrf_err_frames = []

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

    if t_enkf is not None and not np.isclose(t_enkf, times[-1]):
        raise RuntimeError(f"Time mismatch between truth and EnKF at cycle {cyc_t}")
    if t_ensrf is not None and not np.isclose(t_ensrf, times[-1]):
        raise RuntimeError(f"Time mismatch between truth and EnSRF at cycle {cyc_t}")

cycles = np.array(cycles)
times = np.array(times)

metric_cycles = np.atleast_1d(metrics["cycle"]).astype(int)
metric_times = np.atleast_1d(metrics["time"]).astype(float)
enkf_rmse = np.atleast_1d(metrics["enkf_rmse_a"]).astype(float)
ensrf_rmse = np.atleast_1d(metrics["ensrf_rmse_a"]).astype(float)

if not np.array_equal(metric_cycles, cycles):
    raise RuntimeError("Cycle mismatch between field files and metrics.csv")

field_min = min(np.min(frame) for frame in truth_frames + enkf_frames + ensrf_frames)
field_max = max(np.max(frame) for frame in truth_frames + enkf_frames + ensrf_frames)

all_err_frames = enkf_err_frames + ensrf_err_frames
err_abs = max(np.max(np.abs(frame)) for frame in all_err_frames)
err_abs = max(err_abs, 1.0e-12)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

ax_truth = axes[0, 0]
ax_enkf = axes[0, 1]
ax_ensrf = axes[0, 2]
ax_enkf_err = axes[1, 0]
ax_ensrf_err = axes[1, 1]
ax_rmse = axes[1, 2]

im_truth = ax_truth.imshow(
    truth_frames[0], origin="lower", cmap="viridis",
    vmin=field_min, vmax=field_max
)
im_enkf = ax_enkf.imshow(
    enkf_frames[0], origin="lower", cmap="viridis",
    vmin=field_min, vmax=field_max
)
im_ensrf = ax_ensrf.imshow(
    ensrf_frames[0], origin="lower", cmap="viridis",
    vmin=field_min, vmax=field_max
)
im_enkf_err = ax_enkf_err.imshow(
    enkf_err_frames[0], origin="lower", cmap="RdBu_r",
    vmin=-err_abs, vmax=err_abs
)
im_ensrf_err = ax_ensrf_err.imshow(
    ensrf_err_frames[0], origin="lower", cmap="RdBu_r",
    vmin=-err_abs, vmax=err_abs
)

fig.colorbar(im_truth, ax=ax_truth, fraction=0.046, pad=0.04)
fig.colorbar(im_enkf, ax=ax_enkf, fraction=0.046, pad=0.04)
fig.colorbar(im_ensrf, ax=ax_ensrf, fraction=0.046, pad=0.04)
fig.colorbar(im_enkf_err, ax=ax_enkf_err, fraction=0.046, pad=0.04)
fig.colorbar(im_ensrf_err, ax=ax_ensrf_err, fraction=0.046, pad=0.04)

ax_truth.set_title("Truth")
ax_enkf.set_title("Stochastic EnKF Mean")
ax_ensrf.set_title("EnSRF Mean")
ax_enkf_err.set_title("EnKF Mean - Truth")
ax_ensrf_err.set_title("EnSRF Mean - Truth")

for axis in [ax_truth, ax_enkf, ax_ensrf, ax_enkf_err, ax_ensrf_err]:
    axis.set_xlabel("i")
    axis.set_ylabel("j")

(line_enkf,) = ax_rmse.plot(metric_cycles[:1], enkf_rmse[:1], marker="o", label="EnKF RMSE")
(line_ensrf,) = ax_rmse.plot(metric_cycles[:1], ensrf_rmse[:1], marker="s", label="EnSRF RMSE")
(marker_enkf,) = ax_rmse.plot([metric_cycles[0]], [enkf_rmse[0]], marker="o", color=line_enkf.get_color())
(marker_ensrf,) = ax_rmse.plot([metric_cycles[0]], [ensrf_rmse[0]], marker="s", color=line_ensrf.get_color())

ax_rmse.set_xlim(metric_cycles.min(), metric_cycles.max())
ymax = max(np.max(enkf_rmse), np.max(ensrf_rmse))
ax_rmse.set_ylim(0.0, ymax * 1.1 if ymax > 0 else 1.0)
ax_rmse.set_xlabel("DA cycle")
ax_rmse.set_ylabel(f"Analysis RMSE of {FIELD_NAME}")
ax_rmse.set_title("RMSE History")
ax_rmse.grid(True)
ax_rmse.legend(loc="upper right")

suptitle = fig.suptitle("")
plt.tight_layout(rect=[0, 0, 1, 0.96])


def update(frame):
    im_truth.set_data(truth_frames[frame])
    im_enkf.set_data(enkf_frames[frame])
    im_ensrf.set_data(ensrf_frames[frame])
    im_enkf_err.set_data(enkf_err_frames[frame])
    im_ensrf_err.set_data(ensrf_err_frames[frame])

    line_enkf.set_data(metric_cycles[: frame + 1], enkf_rmse[: frame + 1])
    line_ensrf.set_data(metric_cycles[: frame + 1], ensrf_rmse[: frame + 1])
    marker_enkf.set_data([metric_cycles[frame]], [enkf_rmse[frame]])
    marker_ensrf.set_data([metric_cycles[frame]], [ensrf_rmse[frame]])

    suptitle.set_text(
        f"{FIELD_NAME} | cycle={cycles[frame]} | time={metric_times[frame]:.5f} | "
        f"EnKF RMSE={enkf_rmse[frame]:.6e} | EnSRF RMSE={ensrf_rmse[frame]:.6e}"
    )

    return (
        im_truth,
        im_enkf,
        im_ensrf,
        im_enkf_err,
        im_ensrf_err,
        line_enkf,
        line_ensrf,
        marker_enkf,
        marker_ensrf,
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
