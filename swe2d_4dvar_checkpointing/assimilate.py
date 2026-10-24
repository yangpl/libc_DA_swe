#!/usr/bin/env python3
"""Reproducible strong-constraint 4D-Var twin experiment; run --help."""
import argparse
import hashlib
import platform
import scipy
import csv
import json
from pathlib import Path
import time
import numpy as np
from scipy.fft import dctn, idctn
from scipy.optimize import minimize
from swe import Model, ModelError


class BackgroundTransform:
    """x = xb + L v, B = L L^T. Symmetric DCT Gaussian square root.

    Sigma is the spatially averaged pointwise standard deviation, not the
    standard deviation of the DCT coefficients. Reflecting covariance edges
    need not match the model's open physical boundary condition.
    """
    def __init__(self, shape, sigma=(.03, .015, .015), length=.06, lx=1., ly=1.):
        _, ny, nx = shape
        ky, kx = np.meshgrid(np.arange(ny)*np.pi/ly, np.arange(nx)*np.pi/lx, indexing='ij')
        kernel = np.exp(-.5 * length**2 * (kx*kx + ky*ky))
        kernel *= np.sqrt(nx*ny / np.sum(kernel**2))
        self.weights = np.array(sigma)[:, None, None] * kernel
        self.shape = shape

    def apply(self, v):
        return idctn(self.weights * dctn(np.asarray(v).reshape(self.shape),
                                       axes=(-2, -1), norm='ortho'),
                     axes=(-2, -1), norm='ortho')

    transpose = apply


class Objective:
    def __init__(self, model, background, transform, indices, observations, sigma):
        self.model, self.background, self.transform = model, background, transform
        self.indices, self.observations, self.sigma = indices, observations, sigma
        self.calls, self.rejected = 0, 0
        self.cached_v = self.cached_value = None
        self.history = []

    def __call__(self, v):
        if self.cached_v is not None and np.array_equal(v, self.cached_v):
            return self.cached_value
        self.calls += 1
        x = self.background + self.transform.apply(v)
        try:
            jo, gx = self.model.evaluate(x, self.indices, self.observations, self.sigma)
        except ModelError as exc:
            if 'CFL' not in str(exc) and 'dry state' not in str(exc):
                raise
            self.rejected += 1
            # L-BFGS-B can backtrack from a nonfinite trial cost. No projection
            # or clipping is used, since that would invalidate this derivative.
            return np.inf, np.zeros_like(v)
        jb = .5 * np.dot(v, v)
        gv = v + self.transform.transpose(gx).ravel()
        self.cached_v = v.copy()
        self.cached_value = (jb + jo, gv)
        self.checkpoint_stats = self.model.last_checkpoint_stats.copy()
        self.parts = (jb, jo)
        return self.cached_value

    def record(self, v):
        cost, grad = self(v)
        jb, jo = self.parts
        row = dict(iteration=len(self.history), cost=float(cost), background_cost=float(jb),
                   observation_cost=float(jo), gradient_norm=float(np.linalg.norm(grad)),
                   gradient_inf=float(np.linalg.norm(grad, ord=np.inf)), evaluations=self.calls)
        self.history.append(row)
        print(f"iter={row['iteration']:3d} J={cost:12.6f} Jb={jb:9.4f} Jo={jo:10.4f} "
              f"|g|inf={row['gradient_inf']:.3e}", flush=True)


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_gradient(objective, seed=321):
    """Check away from xb, so both B and observation derivatives are tested."""
    rng = np.random.default_rng(seed)
    n = objective.background.size
    v = rng.normal(size=n) * .03
    j0, grad = objective(v)
    rows = []
    for direction in range(3):
        d = rng.normal(size=n)
        d /= np.linalg.norm(d)
        adj = float(grad @ d)
        for eps in (1., .3, .1, .03, .01, .003, .001, .0003, .0001):
            # Forward-only evaluations: do not mutate the reference gradient.
            costs = []
            for sign in (1, -1):
                trial = v + sign*eps*d
                jo, _ = objective.model.evaluate(objective.background + objective.transform.apply(trial),
                         objective.indices, objective.observations, objective.sigma, gradient=False)
                costs.append(.5*np.dot(trial, trial) + jo)
            fd = (costs[0] - costs[1])/(2*eps)
            rows.append(dict(direction=direction, epsilon=eps, adjoint=adj, finite_difference=fd,
                             relative_error=abs(fd-adj)/max(abs(adj), abs(fd), 1e-8),
                             taylor_residual=abs(costs[0]-j0-eps*adj)))
    best = [min(r['relative_error'] for r in rows if r['direction'] == d) for d in range(3)]
    # A wrong sign must fail: never take absolute values of the derivative ratio.
    if max(best) > 2e-5:
        raise RuntimeError(f'Adjoint directional checks failed: {best}')
    return rows, best


def experiment(args):
    validate_args(args)
    started = time.perf_counter()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = Model(args.nx, args.ny, args.dt, args.stride, args.cycles,
                  checkpoint_budget=args.checkpoint_budget)
    rng_bg, rng_obs, rng_hold = [np.random.default_rng(args.seed + i) for i in range(3)]
    x = (np.arange(args.nx)+.5)/args.nx
    y = (np.arange(args.ny)+.5)/args.ny
    xx, yy = np.meshgrid(x, y)
    truth0 = np.zeros(model.shape)
    truth0[0] = 1 + .2*np.exp(-((xx-.5)**2 + (yy-.5)**2)/.02)
    transform = BackgroundTransform(model.shape, (args.sigma_h, args.sigma_m, args.sigma_m), args.length)
    # A draw from the stated B: spatially coherent uncertainty remains observable
    # after numerical diffusion, unlike independent grid-cell white noise.
    background0 = truth0 + transform.apply(rng_bg.normal(size=model.shape))
    forecast = Model(args.nx, args.ny, args.dt, args.stride, args.cycles + args.forecast_cycles)
    truth, diag_truth = forecast.forward(truth0)
    background, diag_bg = forecast.forward(background0)
    axis_x = np.linspace(args.nx//8, args.nx-1-args.nx//8, args.sensors_side).astype(int)
    axis_y = np.linspace(args.ny//8, args.ny-1-args.ny//8, args.sensors_side).astype(int)
    if len(np.unique(axis_x)) < len(axis_x) or len(np.unique(axis_y)) < len(axis_y):
        raise ValueError('Too many sensors for the grid')
    iy, ix = np.meshgrid(axis_y, axis_x, indexing='ij')
    indices = (iy*args.nx+ix).ravel().astype(np.int32)
    candidates = np.setdiff1d(np.arange(args.nx*args.ny), indices)
    hold_indices = rng_hold.choice(candidates, size=min(len(indices),len(candidates)), replace=False)
    clean = truth[1:args.cycles+1, 0].reshape(args.cycles, -1)[:, indices]
    obs = clean + args.obs_std * rng_obs.normal(size=clean.shape)
    held_obs = truth[1:args.cycles+1, 0].reshape(args.cycles, -1)[:, hold_indices]
    held_obs = held_obs + args.obs_std * rng_hold.normal(size=held_obs.shape)
    objective = Objective(model, background0, transform, indices, obs, args.obs_std)
    print(f'Grid {args.nx}x{args.ny}; {len(indices)} height sensors; '
          f'assimilation t={args.dt*args.stride*args.cycles:.3f}; '
          f'forecast t={args.dt*args.stride*args.forecast_cycles:.3f}', flush=True)
    checks, best = check_gradient(objective)
    write_csv(out/'gradient_check.csv', checks)
    print(f'Gradient check best relative errors: {best}', flush=True)
    objective.calls = objective.rejected = 0
    v0 = np.zeros(background0.size)
    objective.record(v0)
    result = minimize(objective, v0, jac=True, method='L-BFGS-B', callback=objective.record,
                      options=dict(maxiter=args.maxiter, maxcor=12, gtol=1e-5, ftol=1e-12, maxls=30))
    if not np.array_equal(result.x, objective.cached_v):
        objective.record(result.x)
    analysis0 = background0 + transform.apply(result.x)
    analysis, diag_an = forecast.forward(analysis0)
    write_csv(out/'cost_history.csv', objective.history)
    times = np.arange(len(truth))*args.dt*args.stride
    rows = []
    for k, t in enumerate(times):
        row = dict(cycle=k, time=float(t), assimilated=int(0 < k <= args.cycles))
        for name, field in [('background', background), ('analysis', analysis)]:
            for component, c in [('h',0), ('hu',1), ('hv',2)]:
                row[f'{name}_rmse_{component}'] = float(np.sqrt(np.mean((field[k,c]-truth[k,c])**2)))
            row[f'{name}_mass'] = float(field[k,0].mean())
        row['truth_mass'] = float(truth[k,0].mean())
        rows.append(row)
    write_csv(out/'trajectory_metrics.csv', rows)
    def obs_rmse(traj, idx, observations):
        pred = traj[1:args.cycles+1, 0].reshape(args.cycles,-1)[:,idx]
        return float(np.sqrt(np.mean((pred-observations)**2)))
    def window_rmse(traj, sl):
        return np.sqrt(np.mean((traj[sl]-truth[sl])**2,axis=(0,2,3))).tolist()
    summary = dict(config=vars(args), optimizer=dict(success=bool(result.success), message=str(result.message),
                   iterations=int(result.nit), evaluations=objective.calls, rejected_trials=objective.rejected),
                   adjoint=dict(backend='Manual discrete adjoint / binomial checkpointing',
                                checkpoint_budget=args.checkpoint_budget,
                                **objective.checkpoint_stats),
                   initial_cost=objective.history[0]['cost'], final_cost=float(result.fun),
                   final_gradient_inf=float(np.max(np.abs(result.jac))),
                   gradient_check_best_relative_errors=best,
                   assimilation_rmse=dict(background=window_rmse(background,slice(0,args.cycles+1)),
                                          analysis=window_rmse(analysis,slice(0,args.cycles+1))),
                   forecast_rmse=dict(background=window_rmse(background,slice(args.cycles+1,None)),
                                      analysis=window_rmse(analysis,slice(args.cycles+1,None))),
                   training_obs_rmse=dict(background=obs_rmse(background,indices,obs),analysis=obs_rmse(analysis,indices,obs)),
                   held_out_obs_rmse=dict(background=obs_rmse(background,hold_indices,held_obs),analysis=obs_rmse(analysis,hold_indices,held_obs)),
                   diagnostics=dict(truth=diag_truth,background=diag_bg,analysis=diag_an),
                   elapsed_seconds=time.perf_counter()-started,
                   environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                       source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                      for name in ('swe2d_4dvar.c','swe2d_4dvar.h','swe.py','assimilate.py',
                                                   'Makefile','requirements.txt')}))
    np.savez_compressed(out/'experiment.npz', x=x,y=y,times=times,truth=truth,background=background,
                        analysis=analysis,obs_indices=indices,observations=obs,held_out_indices=hold_indices,
                        held_out_observations=held_obs,control=result.x)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if not result.success:
        raise RuntimeError(f'Optimization did not converge: {result.message}; diagnostics saved in {out}')
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='var_output')
    p.add_argument('--nx', type=int, default=60)
    p.add_argument('--ny', type=int, default=60)
    p.add_argument('--dt', type=float, default=.001)
    p.add_argument('--stride', type=int, default=20, help='Steps between observations')
    p.add_argument('--checkpoint-budget', type=int, default=8,
                   help='Binomial state checkpoint slots, independent of observations (1..90)')
    p.add_argument('--cycles', type=int, default=15, help='Observation times in one 4D-Var window')
    p.add_argument('--forecast-cycles', type=int, default=10, help='Unassimilated forecast times')
    p.add_argument('--sensors-side', type=int, default=6)
    p.add_argument('--obs-std', type=float, default=.003)
    p.add_argument('--sigma-h', type=float, default=.03)
    p.add_argument('--sigma-m', type=float, default=.015)
    p.add_argument('--length', type=float, default=.06, help='Background square-root smoothing length')
    p.add_argument('--maxiter', type=int, default=400)
    p.add_argument('--seed', type=int, default=20261016)
    return p


def validate_args(args):
    if not 1 <= args.checkpoint_budget <= 90:
        raise ValueError('Checkpoint budget must be from 1 to 90')
    if (args.forecast_cycles < 1 or args.sensors_side < 1 or args.maxiter < 1 or
        min(args.obs_std,args.sigma_h,args.sigma_m) <= 0 or args.length < 0 or
        not np.all(np.isfinite([args.obs_std,args.sigma_h,args.sigma_m,args.length]))):
        raise ValueError('Invalid experiment parameters')
    if (args.nx < 2 or args.ny < 2 or
        args.sensors_side > min(args.nx-2*(args.nx//8), args.ny-2*(args.ny//8))):
        raise ValueError('Grid must be at least 2x2 with distinct sensor locations')
    if args.sensors_side**2 >= args.nx*args.ny:
        raise ValueError('Leave at least one grid cell for held-out validation')


if __name__ == '__main__':
    cli = parser()
    args = cli.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        cli.error(str(exc))
    experiment(args)
