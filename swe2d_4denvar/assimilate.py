#!/usr/bin/env python3
"""Reproducible SWE 4D-EnVar twin experiment with a subsequent free forecast."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from covariance import BackgroundTransform
from envar import EnsembleProblem, make_ensemble
from swe import Model, ModelError


def write_csv(path, rows):
    with open(path, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_gradient(objective, seed=321):
    """Finite differences of the fixed-ensemble QUADRATIC, not nonlinear SWE."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=objective.size)*.03
    j, g = objective(w)
    rows = []
    for direction in range(3):
        d = rng.normal(size=w.size)
        d /= np.linalg.norm(d)
        derivative = float(g @ d)
        for eps in (1., .1, .01, .001, .0001, .00001):
            jp = objective(w+eps*d)[0]
            jm = objective(w-eps*d)[0]
            fd = (jp-jm)/(2*eps)
            rows.append(dict(direction=direction, epsilon=eps, analytic=derivative,
                             finite_difference=fd,
                             relative_error=abs(fd-derivative)/max(1., abs(fd), abs(derivative)),
                             taylor_residual=abs(jp-j-eps*derivative)))
    best = [min(r['relative_error'] for r in rows if r['direction'] == i) for i in range(3)]
    if max(best) > 1e-7:
        raise RuntimeError(f'Ensemble-space gradient check failed: {best}')
    return rows, best


def experiment(args):
    validate_args(args)
    started = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(args.nx, args.ny, args.dt, args.stride, args.cycles)
    forecast = Model(args.nx, args.ny, args.dt, args.stride, args.cycles+args.forecast_cycles)
    x, y = (np.arange(args.nx)+.5)/args.nx, (np.arange(args.ny)+.5)/args.ny
    xx, yy = np.meshgrid(x, y)
    truth0 = np.zeros(model.shape)
    truth0[0] = 1+.2*np.exp(-((xx-.5)**2+(yy-.5)**2)/.02)
    transform = BackgroundTransform(model.shape, (args.sigma_h, args.sigma_m, args.sigma_m), args.length)
    rng_bg, rng_obs, rng_hold = [np.random.default_rng(args.seed+i) for i in range(3)]
    background0 = truth0 + transform.apply(rng_bg.normal(size=model.shape))
    truth, diag_truth = forecast.forward(truth0)
    background, diag_bg = forecast.forward(background0)
    ax = np.linspace(args.nx//8, args.nx-1-args.nx//8, args.sensors_side).astype(int)
    ay = np.linspace(args.ny//8, args.ny-1-args.ny//8, args.sensors_side).astype(int)
    iy, ix = np.meshgrid(ay, ax, indexing='ij')
    indices = (iy*args.nx+ix).ravel().astype(np.int32)
    candidates = np.setdiff1d(np.arange(args.nx*args.ny), indices)
    hold = rng_hold.choice(candidates, size=min(len(indices), len(candidates)), replace=False)
    clean = truth[1:args.cycles+1, 0].reshape(args.cycles, -1)
    observations = clean[:, indices] + args.obs_std*rng_obs.normal(size=(args.cycles, len(indices)))
    held_obs = clean[:, hold] + args.obs_std*rng_hold.normal(size=(args.cycles, len(hold)))
    members = make_ensemble(background0, transform, args.members, args.ensemble_seed, args.spread)
    print(f'Grid {args.nx}x{args.ny}; {args.members} members; {len(indices)} sensors; '
          f'{args.cycles} observation times; solver={args.solver}', flush=True)
    problem = EnsembleProblem(model, background0, members, observations, args.obs_std, indices=indices)
    checks, best = check_gradient(problem.objective)
    write_csv(out/'gradient_check.csv', checks)
    coefficients, solver = problem.objective.solve(args.solver, args.rtol, args.maxiter)
    write_csv(out/'cost_history.csv', solver['history'])
    analysis0 = problem.initial_state(coefficients)
    # Always integrate the recovered initial state. A linear combination of
    # ensemble trajectories is generally not a nonlinear SWE trajectory.
    try:
        analysis, diag_an = forecast.forward(analysis0)
    except ModelError as exc:
        (out/'failure.json').write_text(json.dumps(dict(error=str(exc), config=vars(args),
                solver=solver, message='Recovered initial state failed nonlinear validation; no clipping applied.'), indent=2)+'\n')
        raise
    predicted = model.observe(analysis[:args.cycles+1], indices=indices)
    surrogate = problem.predicted_observations(coefficients)
    jb = float(.5*(coefficients @ coefficients))
    nonlinear_cost = jb + .5*float(np.sum(((predicted-observations)/args.obs_std)**2))
    times = np.arange(len(truth))*args.dt*args.stride
    metrics = []
    for k, t in enumerate(times):
        row = dict(cycle=k, time=float(t), assimilated=int(0 < k <= args.cycles))
        for name, values in [('background', background), ('analysis', analysis)]:
            for c, component in enumerate(('h', 'hu', 'hv')):
                row[f'{name}_rmse_{component}'] = float(np.sqrt(np.mean((values[k, c]-truth[k, c])**2)))
            row[f'{name}_mass'] = float(values[k, 0].mean())
        row['truth_mass'] = float(truth[k, 0].mean())
        metrics.append(row)
    write_csv(out/'trajectory_metrics.csv', metrics)

    def rmse(values, selection):
        return np.sqrt(np.mean((values[selection]-truth[selection])**2, axis=(0, 2, 3))).tolist()

    def obs_rmse(values, locations, obs):
        samples = values[1:args.cycles+1, 0].reshape(args.cycles, -1)[:, locations]
        return float(np.sqrt(np.mean((samples-obs)**2)))

    summary = dict(config=vars(args), method='Fixed-ensemble 4D-EnVar; nonlinear background innovation',
        solver={k: v for k, v in solver.items() if k != 'history'},
        ensemble=dict(size=args.members, maximum_rank=args.members-1,
                      initial_anomaly_rank=int(np.linalg.matrix_rank(problem.x)),
                      window_forward_integrations=args.members+1,
                      anomaly_normalization='1/sqrt(Ne-1)', localization=False,
                      mean_observation_offset_rms=float(np.sqrt(np.mean(
                          (problem.ensemble_mean_observations-problem.background_observations)**2)))),
        initial_cost=solver['history'][0]['cost'], final_surrogate_cost=problem.objective(coefficients)[0],
        nonlinear_analysis_cost=nonlinear_cost,
        nonlinear_cost_reduced=bool(nonlinear_cost < solver['history'][0]['cost']),
        final_gradient_inf=float(np.linalg.norm(problem.objective(coefficients)[1], np.inf)),
        surrogate_observation_error_rms=float(np.sqrt(np.mean((predicted-surrogate)**2))),
        gradient_check_best_relative_errors=best,
        assimilation_rmse={name: rmse(values, slice(0, args.cycles+1))
                           for name, values in [('background', background), ('analysis', analysis)]},
        forecast_rmse={name: rmse(values, slice(args.cycles+1, None))
                       for name, values in [('background', background), ('analysis', analysis)]},
        training_obs_rmse={name: obs_rmse(values, indices, observations)
                           for name, values in [('background', background), ('analysis', analysis)]},
        held_out_obs_rmse={name: obs_rmse(values, hold, held_obs)
                           for name, values in [('background', background), ('analysis', analysis)]},
        diagnostics=dict(truth=diag_truth, background=diag_bg, ensemble=problem.diagnostics, analysis=diag_an),
        elapsed_seconds=time.perf_counter()-started,
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                         source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in Path(__file__).resolve().parent.iterdir()
                                        if p.suffix in ('.c', '.h', '.py') or p.name == 'Makefile'}))
    np.savez_compressed(out/'experiment.npz', x=x, y=y, times=times, truth=truth,
        background=background, analysis=analysis, obs_indices=indices, observations=observations,
        held_out_indices=hold, held_out_observations=held_obs, control=coefficients,
        ensemble_initial=members, ensemble_observations=problem.member_observations,
        background_observations=problem.background_observations,
        surrogate_observations=surrogate, nonlinear_analysis_observations=predicted)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(f'Quadratic cost: {summary["initial_cost"]:.9g} -> {summary["final_surrogate_cost"]:.9g}; '
          f'nonlinear analysis cost: {nonlinear_cost:.9g}', flush=True)
    print(f'Forecast depth RMSE: {summary["forecast_rmse"]["background"][0]:.6g} -> '
          f'{summary["forecast_rmse"]["analysis"][0]:.6g}', flush=True)
    print(f'Results saved in {out.resolve()}', flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='envar_output')
    p.add_argument('--nx', type=int, default=60)
    p.add_argument('--ny', type=int, default=60)
    p.add_argument('--dt', type=float, default=.001)
    p.add_argument('--stride', type=int, default=20)
    p.add_argument('--cycles', type=int, default=15)
    p.add_argument('--forecast-cycles', type=int, default=10)
    p.add_argument('--sensors-side', type=int, default=6)
    p.add_argument('--obs-std', type=float, default=.003)
    p.add_argument('--sigma-h', type=float, default=.03)
    p.add_argument('--sigma-m', type=float, default=.015)
    p.add_argument('--length', type=float, default=.06)
    p.add_argument('--seed', type=int, default=20261016)
    p.add_argument('--ensemble-seed', type=int, default=20261019)
    p.add_argument('--members', type=int, default=80)
    p.add_argument('--spread', type=float, default=1., help='Multiply ensemble perturbations by this factor')
    p.add_argument('--solver', choices=('cholesky', 'cg'), default='cholesky')
    p.add_argument('--rtol', type=float, default=1e-12)
    p.add_argument('--maxiter', type=int, default=None, help='Maximum CG iterations (default: 10*members)')
    return p


def validate_args(a):
    if (min(a.nx, a.ny) < 2 or max(a.nx, a.ny) > 4096 or min(a.stride, a.cycles, a.forecast_cycles) < 1
            or a.stride*(a.cycles+a.forecast_cycles) > 1000000 or a.members < 2
            or a.sensors_side < 1 or a.sensors_side > min(a.nx-2*(a.nx//8), a.ny-2*(a.ny//8))
            or a.sensors_side**2 >= a.nx*a.ny or min(a.seed, a.ensemble_seed) < 0
            or not np.all(np.isfinite([a.dt, a.obs_std, a.sigma_h, a.sigma_m, a.length, a.spread, a.rtol]))
            or min(a.dt, a.obs_std, a.sigma_h, a.sigma_m, a.spread) <= 0 or a.length < 0
            or not 0 < a.rtol < 1 or (a.maxiter is not None and a.maxiter < 1)):
        raise ValueError('Invalid experiment parameters; check grid, schedule, sensors, ensemble size, and positive scales')


if __name__ == '__main__':
    cli = parser()
    args = cli.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        cli.error(str(exc))
    experiment(args)
