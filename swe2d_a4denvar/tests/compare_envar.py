#!/usr/bin/env python3
"""Optional comparison against the sibling 4D-EnVar Cholesky and CG solvers.

Uses identical anomalies, innovations, and R reconstructed from a saved run.
No files in the sibling project are modified.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aenvar import QuadraticObjective
from check_saved import saved_objective


def main(output):
    out = Path(output)
    summary, y, d = saved_objective(out)
    reference_path = Path(__file__).resolve().parents[2]/'swe2d_4denvar/envar.py'
    spec = importlib.util.spec_from_file_location('reference_envar', reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    sigma = summary['config']['obs_std']
    w = QuadraticObjective(y, d, sigma).solve('ensemble')[0]
    oracle = reference.QuadraticObjective(y, d, sigma)
    direct = oracle.solve('cholesky')[0]
    iterative, cg_info = oracle.solve('cg')
    np.testing.assert_array_equal(w, direct)
    np.testing.assert_allclose(w, iterative, rtol=1e-8, atol=2e-9)
    with np.load(out/'experiment.npz') as data:
        members = data['ensemble_initial']
        x = (members-members.mean(axis=0)).reshape(len(members), -1).T/np.sqrt(len(members)-1)
        state_delta = x @ (w-iterative)
    result = dict(original_cholesky_coefficients_identical=True,
                  original_cg_iterations=cg_info['iterations'],
                  cg_coefficient_max_difference=float(np.max(np.abs(w-iterative))),
                  cg_coefficient_relative_l2_difference=float(np.linalg.norm(w-iterative)/max(1., np.linalg.norm(w))),
                  cg_initial_state_max_difference=float(np.max(np.abs(state_delta))))
    (out/'envar_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='a4denvar_output')
    main(parser.parse_args().output)
