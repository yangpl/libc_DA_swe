"""Binomial replay invariance and safe concurrent calls through the public API."""
from concurrent.futures import ThreadPoolExecutor
import unittest
import numpy as np
from swe import Model


class CheckpointingTests(unittest.TestCase):
    def test_budgets_and_concurrent_calls(self):
        rng = np.random.default_rng(723)
        x = rng.normal(size=(3, 7, 9))*.02
        x[0] += 1.
        saved = x.copy()
        indices = np.array([0, 8, 54, 62, 31], dtype=np.int32)
        obs = 1.+rng.normal(size=(7, 5))*.01
        def evaluate(budget):
            m = Model(9, 7, stride=3, cycles=7, checkpoint_budget=budget)
            cost, g = m.evaluate(x, indices, obs, .02)
            return cost, g
        # Each model shares the same C runtime; C mutex must protect the tape.
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(evaluate, [1, 2, 4, 8, 1, 2, 4, 8]))
        for j, g in results:
            self.assertEqual(j, results[0][0])
            np.testing.assert_allclose(g, results[0][1], atol=2e-12, rtol=2e-12)
        np.testing.assert_array_equal(x, saved)

    def test_invalid_budgets(self):
        for budget in (-1, 0, 91, 2.5):
            with self.assertRaises(ValueError):
                Model(checkpoint_budget=budget)
