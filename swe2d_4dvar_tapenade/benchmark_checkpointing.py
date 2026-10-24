#!/usr/bin/env python3
"""Compare generated gradients and standalone binomial schedules with the manual oracle."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from swe import Model
from assimilate import write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='var_output')
    parser.add_argument('--output', default='var_output_tapenade')
    args = parser.parse_args()
    subprocess.run(['make', '-s', 'build/checkpoint_schedule'], cwd=ROOT, check=True)
    ref = Path(args.reference)
    cfg = json.loads((ref/'summary.json').read_text())['config']
    data = np.load(ref/'experiment.npz')
    model = Model(cfg['nx'],cfg['ny'],cfg['dt'],cfg['stride'],cfg['cycles'])
    indices = np.ascontiguousarray(data['obs_indices'], dtype=np.int32)
    obs = np.ascontiguousarray(data['observations'])
    rows = []
    with tempfile.TemporaryDirectory(prefix='swe-oracle-') as tmp:
        libpath = Path(tmp)/'reference.so'
        subprocess.run(['gcc','-O3','-std=c11','-shared','-fPIC',
                        str(ROOT/'tests/reference/manual_swe.c'),'-lm','-o',str(libpath)],check=True)
        oracle = ct.CDLL(str(libpath)).swe_evaluate
        oracle.argtypes = model.lib.swe_evaluate.argtypes
        oracle.restype = ct.c_int
        for state_name in ('background','analysis'):
            state = np.ascontiguousarray(data[state_name][0])
            jref = np.empty(1);gref = np.empty_like(state);diag = np.empty(2)
            start = time.perf_counter()
            status = oracle(*model._args(state),len(indices),indices,obs,cfg['obs_std'],jref,gref.ctypes.data,diag)
            if status: raise RuntimeError(f'Manual oracle failed: {status}')
            oracle_time = time.perf_counter()-start
            for budget in (1,2,4,8,16):
                model.checkpoint_budget = budget
                start = time.perf_counter()
                cost, grad = model.evaluate(state,indices,obs,cfg['obs_std'])
                elapsed = time.perf_counter()-start
                error = np.linalg.norm(grad-gref)/np.linalg.norm(gref)
                # Run the official scheduler independently; production AD is uninstrumented.
                stats = json.loads(subprocess.check_output([
                    str(ROOT/'build/checkpoint_schedule'),
                    str(cfg['stride']*cfg['cycles']),str(budget)],text=True))
                if error>2e-12 or cost!=jref[0]:
                    raise RuntimeError(f'Parity failure: {state_name}, {budget}, {error}')
                rows.append(dict(state=state_name,budget=budget,cost=cost,
                    relative_gradient_error=float(error),elapsed_seconds=elapsed,
                    manual_seconds=oracle_time,schedule_source='standalone adBinomial',**stats,
                    snapshot_payload_bytes=stats['peak_snapshots']*(3*state.nbytes+12)))
                print(json.dumps(rows[-1]),flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'checkpoint_benchmark.csv',rows)
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/swe-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bg=[r for r in rows if r['state']=='background']
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    budgets=[r['budget'] for r in bg]
    for ax,key,title,ylabel in zip(axes,
            ('forward_steps','snapshot_payload_bytes','relative_gradient_error'),
            ('Recomputation decreases','Snapshot storage increases','Gradient is unchanged'),
            ('Scheduled forward RK2 steps','Snapshot payload (MB)','Relative error vs manual adjoint')):
        values=[r[key]/(1e6 if key=='snapshot_payload_bytes' else 1) for r in bg]
        ax.plot(budgets,values,'o-',color='#007f86')
        ax.set(xlabel='Checkpoint budget',ylabel=ylabel,title=title,xticks=budgets)
        if key!='snapshot_payload_bytes': ax.set_yscale('log')
        ax.grid(alpha=.2)
    fig.suptitle(f'Tapenade native binomial checkpointing: {cfg["nx"]}×{cfg["ny"]}, {cfg["stride"]*cfg["cycles"]} steps')
    for ext in ('png','pdf'): fig.savefig(out/f'checkpoint_tradeoff.{ext}',dpi=180)
    plt.close(fig)


if __name__ == '__main__': main()
