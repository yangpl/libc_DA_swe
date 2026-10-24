#!/usr/bin/env python3
"""Compare binomial budgets with the frozen block adjoint on the same state."""
import argparse
import csv
import ctypes as ct
from pathlib import Path
import statistics
import subprocess
import time
import numpy as np
from swe import Model, ROOT


def benchmark(args):
    if args.repeat < 1:
        raise ValueError('repeat must be positive')
    model = Model(nx=args.nx, ny=args.ny, stride=args.stride, cycles=args.cycles)
    subprocess.run(['make', '-s', 'build/libblock_reference.so'], cwd=ROOT, check=True)
    reference = ct.CDLL(str(ROOT/'build/libblock_reference.so'))
    reference.swe_evaluate.argtypes = model.lib.swe_evaluate.argtypes
    reference.swe_evaluate.restype = ct.c_int
    rng = np.random.default_rng(47)
    initial = rng.normal(size=model.shape)*.01
    initial[0] += 1.
    indices = np.linspace(0, args.nx*args.ny-1, min(36,args.nx*args.ny), dtype=np.int32)
    obs = np.ascontiguousarray(1.+rng.normal(size=(args.cycles,len(indices)))*.003)

    def old_evaluate():
        cost,diag,grad = np.empty(1),np.empty(2),np.empty(model.shape)
        model._check(reference.swe_evaluate(*model._args(initial),len(indices),indices,
                     obs,.003,cost,grad.ctypes.data,diag))
        return float(cost[0]),grad

    def timed(evaluate):
        durations = []
        evaluate()  # Warm caches and library; not included in the measurements.
        for _ in range(args.repeat):
            started = time.perf_counter()
            cost,grad = evaluate()
            durations.append(time.perf_counter()-started)
        return cost,grad,statistics.median(durations)

    cost_ref,gradient_ref,seconds = timed(old_evaluate)
    ns = args.stride*args.cycles
    rows = [dict(method='block',budget=0,
                 state_storage_bytes=(args.cycles+2*args.stride+7)*initial.nbytes,
                 forward_steps=2*ns,stage_recomputations=0,seconds=seconds,
                 max_gradient_difference=0.)]
    for budget in args.budgets:
        candidate = Model(nx=args.nx,ny=args.ny,stride=args.stride,cycles=args.cycles,
                          checkpoint_budget=budget)
        cost,grad,seconds = timed(lambda: candidate.evaluate(initial,indices,obs,.003))
        np.testing.assert_allclose(cost,cost_ref,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(grad,gradient_ref,rtol=1e-12,atol=1e-10)
        stats = candidate.last_checkpoint_stats
        rows.append(dict(method='binomial',budget=budget,
                         state_storage_bytes=stats['state_storage_bytes'],
                         forward_steps=stats['forward_steps'],
                         stage_recomputations=stats['stage_recomputations'],seconds=seconds,
                         max_gradient_difference=float(np.max(np.abs(grad-gradient_ref)))))
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'checkpoint_benchmark.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'{args.nx} x {args.ny}, {ns} steps; median of {args.repeat} evaluations')
    for row in rows:
        print(f"{row['method']:8s} budget={row['budget']:2d} "
              f"states={row['state_storage_bytes']/1e6:.4f} MB "
              f"RK2={row['forward_steps']:6d} stage={row['stage_recomputations']:4d} "
              f"time={row['seconds']:.4f} s gradient error={row['max_gradient_difference']:.3e}")
    print(f'Saved {out / "checkpoint_benchmark.csv"}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx',type=int,default=60)
    parser.add_argument('--ny',type=int,default=60)
    parser.add_argument('--stride',type=int,default=20)
    parser.add_argument('--cycles',type=int,default=15)
    parser.add_argument('--budgets',type=int,nargs='+',default=[2,4,8,16,32,56])
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--output',default='var_output_checkpoint_benchmark')
    benchmark(parser.parse_args())
