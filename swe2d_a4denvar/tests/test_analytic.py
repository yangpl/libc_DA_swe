"""Analytic ensemble/observation-space identities and automatic selection."""
import unittest
import numpy as np
from aenvar import QuadraticObjective


class AnalyticTests(unittest.TestCase):
    def test_auto_selects_smaller_system_and_matches_augmented_least_squares(self):
        rng = np.random.default_rng(912)
        for nobs, ne in ((3, 11), (15, 5), (6, 6), (1, 2)):
            y = rng.normal(size=(nobs, ne))
            y -= y.mean(axis=1, keepdims=True)
            d = rng.normal(size=nobs)
            sigma = np.linspace(.1, .8, nobs)
            objective = QuadraticObjective(y, d, sigma)
            # Independent rectangular least-squares oracle: no normal equations.
            design = np.vstack([np.eye(ne), y/sigma[:, None]])
            rhs = np.concatenate([np.zeros(ne), d/sigma])
            expected = np.linalg.lstsq(design, rhs, rcond=None)[0]
            for space in ('auto', 'ensemble', 'observation'):
                w, info = objective.solve(space)
                np.testing.assert_allclose(w, expected, rtol=5e-12, atol=5e-13)
                self.assertEqual(info['iterations'], 0)
                self.assertEqual(info['factorizations'], 1)
                self.assertLess(info['relative_residual'], 1e-11)
                if space == 'auto':
                    self.assertEqual(info['system_dimension'], min(ne, nobs))
                    self.assertEqual(info['solve_space'], 'ensemble' if ne <= nobs else 'observation')

    def test_invariance_under_member_and_observation_permutations(self):
        rng = np.random.default_rng(138)
        y = rng.normal(size=(13, 8)); y -= y.mean(axis=1, keepdims=True)
        d, sigma = rng.normal(size=13), np.linspace(.2, 1.3, 13)
        reference = QuadraticObjective(y, d, sigma).solve('ensemble')[0]
        perm_m, perm_o = rng.permutation(8), rng.permutation(13)
        for space in ('ensemble', 'observation'):
            w, _ = QuadraticObjective(y[perm_o][:, perm_m], d[perm_o], sigma[perm_o]).solve(space)
            np.testing.assert_allclose(w, reference[perm_m], rtol=2e-12, atol=2e-13)

    def test_repeated_observations_equal_halving_the_variance(self):
        rng = np.random.default_rng(82)
        y, d = rng.normal(size=(7, 4)), rng.normal(size=7)
        repeated = QuadraticObjective(np.tile(y, (2, 1)), np.tile(d, 2), .3)
        equivalent = QuadraticObjective(y, d, .3/np.sqrt(2))
        for space in ('ensemble', 'observation'):
            np.testing.assert_allclose(repeated.solve(space)[0], equivalent.solve(space)[0], atol=1e-13)


if __name__ == '__main__':
    unittest.main()
