"""Independent covariance oracle, normalization, derivatives, and solver tests."""
import unittest
import numpy as np
from covariance import BackgroundTransform
from aenvar import QuadraticObjective, EnsembleProblem, make_ensemble


class QuadraticTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_both_solvers_match_dense_observation_space_update(self):
        # A stacked linear map represents several distinct observation times.
        n, ne, m = 12, 8, 19
        members = self.rng.normal(size=(ne, n))
        x = (members-members.mean(axis=0)).T/np.sqrt(ne-1)
        h = self.rng.normal(size=(m, n))
        y = h @ x
        d = self.rng.normal(size=m)
        sigma = np.linspace(.3, .7, m)
        b = np.cov(members, rowvar=False, ddof=1)
        expected = b @ h.T @ np.linalg.solve(h @ b @ h.T+np.diag(sigma**2), d)
        objective = QuadraticObjective(y, d, sigma)
        solutions = []
        for method in ('ensemble', 'observation', 'auto'):
            w, info = objective.solve(method)
            np.testing.assert_allclose(x @ w, expected, rtol=2e-11, atol=2e-12)
            self.assertLess(abs(w.sum()), 1e-11)
            self.assertLess(objective(w)[0], objective(np.zeros(ne))[0])
            self.assertLess(info['relative_residual'], 1e-10)
            solutions.append(w)
        for solution in solutions[1:]:
            np.testing.assert_allclose(solutions[0], solution, rtol=1e-11, atol=1e-12)

    def test_gradient_hessian_and_taylor_remainder(self):
        y, d = self.rng.normal(size=(17, 6)), self.rng.normal(size=17)
        objective = QuadraticObjective(y, d, .7)
        w, p = self.rng.normal(size=(2, 6))
        j, g = objective(w)
        eps = 1e-5
        jp, gp = objective(w+eps*p)
        jm, gm = objective(w-eps*p)
        self.assertAlmostEqual((jp-jm)/(2*eps), g @ p, delta=2e-8)
        np.testing.assert_allclose((gp-gm)/(2*eps), objective.hessian_product(p), rtol=2e-10)
        for eps in (.1, .03):
            remainder = objective(w+eps*p)[0]-j-eps*(g @ p)
            self.assertAlmostEqual(remainder, .5*eps**2*(p @ objective.hessian_product(p)), delta=1e-13)

    def test_rank_deficient_and_zero_innovation(self):
        for y in (np.zeros((5, 3)), np.ones((5, 3))):
            for method in ('ensemble', 'observation', 'auto'):
                w, _ = QuadraticObjective(y, np.zeros(5), 1.).solve(method)
                np.testing.assert_array_equal(w, np.zeros(3))
                w, _ = QuadraticObjective(y, np.ones(5), 1.).solve(method)
                self.assertTrue(np.all(np.isfinite(w)))

    def test_invalid_inputs(self):
        y = self.rng.normal(size=(20, 10))
        d = np.ones(20)
        for sigma in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError): QuadraticObjective(y, d, sigma)
        with self.assertRaises(ValueError): QuadraticObjective(y, np.ones(19), 1)
        with self.assertRaises(ValueError): QuadraticObjective(y, d, np.ones(3))
        objective = QuadraticObjective(y, d, 1)
        with self.assertRaises(ValueError): objective(np.ones(9))
        with self.assertRaises(ValueError): objective.solve('unknown')

    def test_ensemble_centering_covariance_and_reproducibility(self):
        shape, ne, seed = (3, 5, 6), 9, 71
        background = np.ones(shape)
        transform = BackgroundTransform(shape)
        members = make_ensemble(background, transform, ne, seed, spread=1.3)
        np.testing.assert_allclose(members.mean(axis=0), background, atol=5e-16, rtol=0)
        np.testing.assert_array_equal(members, make_ensemble(background, transform, ne, seed, 1.3))
        rng = np.random.default_rng(seed)
        raw = np.array([1.3*transform.apply(rng.normal(size=shape)) for _ in range(ne)]).reshape(ne, -1)
        x = (members-background).reshape(ne, -1).T/np.sqrt(ne-1)
        np.testing.assert_allclose(x @ x.T, np.cov(raw, rowvar=False), rtol=1e-11, atol=1e-17)
        for size in (0, 1, 2.5):
            with self.assertRaises(ValueError): make_ensemble(background, transform, size)


class NonlinearToyModel:
    """Independent multitime nonlinear observation map to catch centering mistakes."""
    shape = (3, 2, 2)

    def forward(self, initial):
        return np.array([initial, initial**2, initial**3]), dict(min_depth=1., max_cfl=.1)

    def observe(self, trajectory, **locations):
        return trajectory[1:, 0].reshape(2, -1)[:, [0, 3]]


class CenteringTests(unittest.TestCase):
    def test_background_innovation_is_not_ensemble_mean(self):
        model = NonlinearToyModel()
        background = np.ones(model.shape)
        members = np.array([background-.2, background, background+.2])
        observations = np.array([[1.1, 1.05], [1.08, 1.1]])
        problem = EnsembleProblem(model, background, members, observations, .1, indices=[0, 3])
        np.testing.assert_array_equal(problem.innovation, (observations-1.).ravel())
        self.assertGreater(np.linalg.norm(problem.ensemble_mean_observations-1.), .01)
        np.testing.assert_allclose(problem.y.sum(axis=1), 0., atol=1e-15)
        w, _ = problem.objective.solve()
        np.testing.assert_allclose(problem.initial_state(w), background+(problem.x @ w).reshape(model.shape))
        actual = model.observe(model.forward(problem.initial_state(w))[0])
        self.assertGreater(np.linalg.norm(actual-problem.predicted_observations(w)), 1e-6)

    def test_uncentered_ensemble_rejected(self):
        model = NonlinearToyModel()
        background = np.ones(model.shape)
        with self.assertRaisesRegex(ValueError, 'mean'):
            EnsembleProblem(model, background, np.array([background, background+.1]),
                            np.ones((2, 2)), .1, indices=[0, 3])


if __name__ == '__main__':
    unittest.main()
