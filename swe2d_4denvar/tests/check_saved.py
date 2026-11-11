#!/usr/bin/env python3
"""Rebuild a saved ensemble quadratic and verify both solvers and nonlinear output."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envar import QuadraticObjective
from swe import Model


def check_saved(output):
    out = Path(output)
    summary = json.loads((out/'summary.json').read_text())
    cfg = summary['config']
    with np.load(out/'experiment.npz') as data:
        ne = cfg['members']
        predictions = data['ensemble_observations']
        anomalies = ((predictions-predictions.mean(axis=0)).reshape(ne, -1).T/np.sqrt(ne-1))
        innovation = (data['observations']-data['background_observations']).ravel()
        objective = QuadraticObjective(anomalies, innovation, cfg['obs_std'])
        direct, info_direct = objective.solve('cholesky')
        iterative, info_cg = objective.solve('cg')
        np.testing.assert_allclose(iterative, direct, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(data['control'], direct, rtol=1e-8, atol=1e-9)
        model = Model(nx=cfg['nx'], ny=cfg['ny'], dt=cfg['dt'], stride=cfg['stride'],
                      cycles=cfg['cycles']+cfg['forecast_cycles'])
        trajectory, _ = model.forward(data['analysis'][0])
        np.testing.assert_array_equal(trajectory, data['analysis'])
        prior = .5*np.sum(data['control']**2)
        residual = (data['nonlinear_analysis_observations']-data['observations'])/cfg['obs_std']
        np.testing.assert_allclose(prior+.5*np.sum(residual**2), summary['nonlinear_analysis_cost'], rtol=1e-13)
        result = dict(cholesky_relative_residual=info_direct['relative_residual'],
                      cg_relative_residual=info_cg['relative_residual'],
                      cg_iterations=info_cg['iterations'],
                      coefficient_max_difference=float(np.max(np.abs(iterative-direct))),
                      coefficient_relative_l2_difference=float(np.linalg.norm(iterative-direct)/max(1., np.linalg.norm(direct))),
                      nonlinear_trajectory_identical=True)
    (out/'solver_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='envar_output')
    check_saved(parser.parse_args().output)
