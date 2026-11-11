"""SWE integration, receiver, and complete experiment regression tests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from assimilate import experiment, parser, validate_args
from covariance import BackgroundTransform
from envar import EnsembleProblem, make_ensemble
from swe import Model, ModelError


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Model(nx=7, ny=5, stride=3, cycles=4, lx=1.3, ly=.9)
        self.initial = np.zeros(self.model.shape)
        self.initial[0], self.initial[1], self.initial[2] = 1.1, .04, -.03

    def test_uniform_flow_and_observation_schedule(self):
        traj, diag = self.model.forward(self.initial)
        np.testing.assert_allclose(traj, np.broadcast_to(self.initial, traj.shape), atol=3e-15, rtol=0)
        self.assertLess(diag['max_cfl'], .9)
        self.assertEqual(self.model.observe(traj, indices=[0, 34]).shape, (4, 2))
        other = Model(nx=7, ny=5, stride=1, cycles=12, lx=1.3, ly=.9)
        state = self.initial.copy()
        state[0] += np.random.default_rng(71).normal(0, .01, (5, 7))
        np.testing.assert_array_equal(self.model.forward(state)[0], other.forward(state)[0][::3])

    def test_receivers_and_index_equivalence(self):
        ix, iy = np.meshgrid(np.arange(7), np.arange(5))
        x, y = (ix+.5)*1.3/7, (iy+.5)*.9/5
        values = 1+2*x-3*y+4*x*y
        receivers = np.array([[.31, .27], [.61, .53], [.31, .27]])
        expected = 1+2*receivers[:, 0]-3*receivers[:, 1]+4*receivers.prod(axis=1)
        np.testing.assert_allclose(self.model.sample_field(values, receivers), expected, atol=1e-14)
        np.testing.assert_array_equal(self.model.sample_field(values, [[0., 0.], [1.3, .9]]),
                                      [values[0, 0], values[-1, -1]])
        traj, _ = self.model.forward(self.initial)
        receivers = np.column_stack([x.ravel(), y.ravel()])
        np.testing.assert_allclose(self.model.observe(traj, receivers=receivers),
                                   self.model.observe(traj, indices=np.arange(35)), atol=1e-15, rtol=0)

    def test_invalid_inputs(self):
        for value in (np.nan, np.inf, 0., -1.):
            bad = self.initial.copy(); bad[0, 0, 0] = value
            with self.assertRaises(ModelError): self.model.forward(bad)
        with self.assertRaisesRegex(ModelError, 'CFL'):
            Model(nx=7, ny=5, dt=.1).forward(self.initial)
        for kwargs in (dict(nx=1), dict(cycles=0), dict(dt=np.inf), dict(stride=1000001)):
            with self.assertRaises(ValueError): Model(**kwargs)
        traj, _ = self.model.forward(self.initial)
        for indices in ([], [1.5], [-1], [35], [2**32]):
            with self.assertRaises(ValueError): self.model.observe(traj, indices=indices)
        for receivers in ([[1.4, .2]], [[np.nan, 0.]], [[0.]], []):
            with self.assertRaises(ValueError): self.model.observe(traj, receivers=receivers)
        with self.assertRaises(ValueError): self.model.observe(traj)
        with self.assertRaises(ValueError): self.model.observe(traj, indices=[0], receivers=[[0., 0.]])

    def test_swe_envar_multitime_repeated_offgrid_receivers(self):
        transform = BackgroundTransform(self.model.shape, sigma=(.002, .001, .001))
        members = make_ensemble(self.initial, transform, 12, seed=13)
        receivers = np.array([[0., 0.], [.31, .27], [.31, .27], [1.3, .9]])
        truth0 = self.initial.copy(); truth0[0] += .001
        observations = self.model.observe(self.model.forward(truth0)[0], receivers=receivers)
        problem = EnsembleProblem(self.model, self.initial, members, observations, .001, receivers=receivers)
        w, _ = problem.objective.solve()
        w_cg, _ = problem.objective.solve('cg')
        np.testing.assert_allclose(w, w_cg, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(problem.y.sum(axis=1), 0., atol=2e-15)
        actual = self.model.observe(self.model.forward(problem.initial_state(w))[0], receivers=receivers)
        self.assertLess(np.linalg.norm(actual-observations), np.linalg.norm(problem.innovation))
        np.testing.assert_array_equal(problem.member_observations[:, :, 1], problem.member_observations[:, :, 2])

    def test_invalid_ensemble_member_identified(self):
        members = np.array([self.initial.copy(), self.initial.copy()])
        members[0, 0, 0, 0] -= 2.
        members[1, 0, 0, 0] += 2.
        with self.assertRaisesRegex(ModelError, 'Ensemble member 0'):
            EnsembleProblem(self.model, self.initial, members, np.ones((4, 1)), .1, indices=[0])


class ExperimentTests(unittest.TestCase):
    def test_complete_twin_experiment_and_saved_nonlinear_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(['--nx', '8', '--ny', '6', '--cycles', '3', '--stride', '2',
                    '--forecast-cycles', '2', '--sensors-side', '2', '--members', '10', '--output', directory])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = experiment(args)
            self.assertTrue(summary['solver']['success'])
            self.assertLess(summary['final_surrogate_cost'], summary['initial_cost'])
            saved = json.loads((Path(directory)/'summary.json').read_text())
            self.assertEqual(saved['config']['members'], 10)
            with np.load(Path(directory)/'experiment.npz') as data:
                trajectory, _ = Model(nx=8, ny=6, cycles=5, stride=2).forward(data['analysis'][0])
                np.testing.assert_array_equal(trajectory, data['analysis'])
                residual = data['analysis'][1:4, 0].reshape(3, -1)[:, data['obs_indices']]-data['observations']
                expected = .5*np.sum(data['control']**2)+.5*np.sum((residual/args.obs_std)**2)
                self.assertAlmostEqual(saved['nonlinear_analysis_cost'], expected, places=10)
                expected_mean = data['ensemble_initial'].mean(axis=0)
                np.testing.assert_allclose(expected_mean, data['background'][0], atol=1e-15)

    def test_invalid_cli_configuration(self):
        for tokens in (['--members', '1'], ['--obs-std', 'nan'], ['--sensors-side', '100'],
                       ['--forecast-cycles', '0'], ['--spread', '-1'], ['--seed', '-1']):
            with self.assertRaises(ValueError): validate_args(parser().parse_args(tokens))


if __name__ == '__main__':
    unittest.main()
