#!/usr/bin/env python3
"""Re-solve a saved experiment in both analytic spaces and verify its trajectory."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aenvar import QuadraticObjective
from swe import Model


def saved_objective(out):
    summary = json.loads((out/'summary.json').read_text())
    with np.load(out/'experiment.npz') as data:
        predictions = data['ensemble_observations']
        ne = len(predictions)
        y = ((predictions-predictions.mean(axis=0)).reshape(ne, -1).T/np.sqrt(ne-1))
        d = (data['observations']-data['background_observations']).ravel()
    return summary, y, d


def check_saved(output):
    out = Path(output)
    summary, y, d = saved_objective(out)
    cfg = summary['config']
    objective = QuadraticObjective(y, d, cfg['obs_std'])
    w_ensemble, info_e = objective.solve('ensemble')
    w_observation, info_o = objective.solve('observation')
    w_auto, info_a = objective.solve('auto')
    np.testing.assert_allclose(w_observation, w_ensemble, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(w_auto, w_ensemble, rtol=1e-9, atol=1e-10)
    with np.load(out/'experiment.npz') as data:
        np.testing.assert_allclose(data['control'], w_ensemble, rtol=1e-9, atol=1e-10)
        model = Model(nx=cfg['nx'], ny=cfg['ny'], dt=cfg['dt'], stride=cfg['stride'],
                      cycles=cfg['cycles']+cfg['forecast_cycles'])
        members = data['ensemble_initial']
        x = (members-members.mean(axis=0)).reshape(len(members), -1).T/np.sqrt(len(members)-1)
        initial_e = data['background'][0]+(x @ w_ensemble).reshape(model.shape)
        initial_o = data['background'][0]+(x @ w_observation).reshape(model.shape)
        np.testing.assert_allclose(initial_e, initial_o, atol=1e-11, rtol=1e-11)
        trajectory, _ = model.forward(data['analysis'][0])
        np.testing.assert_array_equal(trajectory, data['analysis'])
        samples = trajectory[1:cfg['cycles']+1, 0].reshape(cfg['cycles'], -1)[:, data['obs_indices']]
        np.testing.assert_array_equal(samples, data['nonlinear_analysis_observations'])
        residual = (samples-data['observations'])/cfg['obs_std']
        cost = .5*np.sum(data['control']**2)+.5*np.sum(residual**2)
        np.testing.assert_allclose(cost, summary['nonlinear_analysis_cost'], rtol=1e-13)
        result = dict(ensemble_relative_residual=info_e['relative_residual'],
                      observation_relative_residual=info_o['relative_residual'],
                      auto_selected_space=info_a['solve_space'],
                      ensemble_system_dimension=info_e['system_dimension'],
                      observation_system_dimension=info_o['system_dimension'],
                      coefficient_max_difference=float(np.max(np.abs(w_ensemble-w_observation))),
                      coefficient_relative_l2_difference=float(np.linalg.norm(w_ensemble-w_observation)/max(1., np.linalg.norm(w_ensemble))),
                      initial_state_max_difference=float(np.max(np.abs(initial_e-initial_o))),
                      nonlinear_trajectory_identical=True)
    (out/'analytic_check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='a4denvar_output')
    check_saved(parser.parse_args().output)
