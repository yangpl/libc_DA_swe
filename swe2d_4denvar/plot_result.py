#!/usr/bin/env python3
"""Plot the saved nonlinear 4D-EnVar analysis and its quadratic diagnostics."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/swe-envar-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_results(output):
    out = Path(output)
    s = json.loads((out/'summary.json').read_text())
    with np.load(out/'experiment.npz') as d:
        times, truth, bg, an = [d[k] for k in ('times', 'truth', 'background', 'analysis')]
    costs = np.atleast_1d(np.genfromtxt(out/'cost_history.csv', delimiter=',', names=True))
    checks = np.atleast_1d(np.genfromtxt(out/'gradient_check.csv', delimiter=',', names=True))
    end = s['config']['cycles']
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for c, name in enumerate(('h', 'hu', 'hv')):
        ax = axes[0, c]
        for label, states in [('Background', bg), ('Analysis', an)]:
            ax.semilogy(times, np.sqrt(np.mean((states[:, c]-truth[:, c])**2, axis=(1, 2))), label=label)
        ax.axvspan(times[end], times[-1], alpha=.15, color='gray', label='Free forecast')
        ax.set(xlabel='Time', ylabel='RMSE', title=f'Nonlinear trajectory: {name}')
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(costs['iteration'], costs['cost'], 'o-', label='Quadratic surrogate')
    ax.axhline(s['nonlinear_analysis_cost'], ls='--', color='tab:red', label='Actual nonlinear final cost')
    ax.set(xlabel='Linear solve step', ylabel='Cost', title=s['solver']['method'])
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for direction in np.unique(checks['direction']):
        r = checks[checks['direction'] == direction]
        ax.loglog(r['epsilon'], np.maximum(r['relative_error'], 1e-16), 'o-', label=f'Direction {int(direction)+1}')
    ax.set(xlabel='Perturbation', ylabel='FD relative error', title='Ensemble quadratic gradient check')
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    for i, label in enumerate(('background', 'analysis')):
        ax.bar(np.arange(2)+(i-.5)*.35,
               [s[k][label] for k in ('training_obs_rmse', 'held_out_obs_rmse')], .35, label=label)
    ax.set(xticks=[0, 1], xticklabels=['Assimilated', 'Held out'], ylabel='Observation residual RMS')
    ax.legend()
    fig.suptitle(f'4D-EnVar: {s["config"]["members"]} members; one joint assimilation window')
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'envar_metrics.{ext}', dpi=160)
    plt.close(fig)

    frames = [0, end, len(times)-1]
    limit = max(np.max(np.abs(bg[frames, 0]-truth[frames, 0])),
                np.max(np.abs(an[frames, 0]-truth[frames, 0])), 1e-15)
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), layout='constrained')
    for row, k in enumerate(frames):
        for col, (name, values) in enumerate([('Truth', truth[k, 0]),
                    ('Background error', bg[k, 0]-truth[k, 0]), ('Analysis error', an[k, 0]-truth[k, 0])]):
            im = axes[row, col].imshow(values, origin='lower', extent=(0, 1, 0, 1),
                    cmap='viridis' if col == 0 else 'RdBu_r',
                    vmin=None if col == 0 else -limit, vmax=None if col == 0 else limit)
            axes[row, col].set(title=f'{name}, t={times[k]:.3f}', xlabel='x', ylabel='y')
            fig.colorbar(im, ax=axes[row, col], shrink=.75)
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'envar_snapshots.{ext}', dpi=160)
    plt.close(fig)
    print(f'Plots saved in {out.resolve()}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='envar_output')
    plot_results(p.parse_args().output)
