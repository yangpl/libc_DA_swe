"""Numerical regression tests: python3 -m unittest discover -s tests -v."""
import unittest
import numpy as np
from swe import Model, ModelError
from assimilate import BackgroundTransform, Objective


class ShallowWaterTests(unittest.TestCase):
    def setUp(self):
        self.model = Model(nx=11, ny=8, dt=.001, stride=4, cycles=3, lx=1.3, ly=.9)
        rng = np.random.default_rng(47)
        self.state = rng.normal(size=self.model.shape) * .015
        self.state[0] += 1.
        self.indices = np.array([0, 10, 77, 87, 24, 49], dtype=np.int32)
        self.obs = np.ones((3,len(self.indices))) + .01*rng.normal(size=(3,len(self.indices)))

    def test_uniform_rest_and_moving_flow(self):
        for momentum in (0., .08):
            state = np.zeros(self.model.shape)
            state[0],state[1],state[2] = 1.2,momentum,-momentum/2
            traj, diag = self.model.forward(state)
            np.testing.assert_allclose(traj,np.broadcast_to(state,traj.shape),rtol=0,atol=3e-15)
            self.assertLess(diag['max_cfl'],.9)

    def test_reject_invalid_states_and_cfl(self):
        for bad in (np.nan, np.inf, -1., 0.):
            state=self.state.copy();state[0,0,0]=bad
            with self.assertRaises(ModelError): self.model.forward(state)
        fast = Model(nx=11,ny=8,dt=.1,stride=4,cycles=3)
        with self.assertRaisesRegex(ModelError,'CFL'): fast.forward(self.state)
        with self.assertRaises(ValueError): self.model.forward(self.state[0])
        with self.assertRaises(ValueError): self.model.evaluate(self.state,[-1],np.ones((3,1)),.01)

    def test_observation_validation_before_c_conversion(self):
        for indices in ([], [0.5], [2**32], [-1], [88], [[0]]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.model.evaluate(self.state,indices,np.ones((3,1)),.01)
        for sigma in (0., -1., np.nan, np.inf):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                self.model.evaluate(self.state,self.indices,self.obs,sigma)
        obs = self.obs.copy(); obs[0,0] = np.nan
        with self.assertRaises(ValueError):
            self.model.evaluate(self.state,self.indices,obs,.01)

    def test_observation_cost_matches_saved_forward_trajectory(self):
        trajectory, _ = self.model.forward(self.state)
        predicted = trajectory[1:,0].reshape(3,-1)[:,self.indices]
        expected = .5*np.sum(((predicted-self.obs)/.02)**2)
        cost, _ = self.model.evaluate(self.state,self.indices,self.obs,.02)
        self.assertAlmostEqual(cost,expected,places=12)

    def test_multitime_adjoint_all_components_and_boundary(self):
        cost, grad = self.model.evaluate(self.state,self.indices,self.obs,.02)
        cost_only, _ = self.model.evaluate(self.state,self.indices,self.obs,.02,False)
        self.assertEqual(cost,cost_only)
        eps=1e-6
        for c in range(3):
            for j,i in [(0,0),(0,10),(7,0),(7,10),(3,5)]:
                plus,minus=self.state.copy(),self.state.copy()
                plus[c,j,i]+=eps;minus[c,j,i]-=eps
                jp,_=self.model.evaluate(plus,self.indices,self.obs,.02,False)
                jm,_=self.model.evaluate(minus,self.indices,self.obs,.02,False)
                fd=(jp-jm)/(2*eps)
                self.assertLess(abs(fd-grad[c,j,i])/max(1.,abs(fd)),2e-6)

    def test_checkpoint_intervals_equivalent(self):
        # Observation frequency must not affect the binomial replay schedule.
        # Zero-residual extra observations preserve the objective and gradient.
        m2 = Model(nx=11,ny=8,dt=.001,stride=2,cycles=6,lx=1.3,ly=.9)
        trajectory,_=m2.forward(self.state)
        obs2=trajectory[1:,0].reshape(6,-1)[:,self.indices].copy()
        obs2[1::2]=self.obs
        j1,g1=self.model.evaluate(self.state,self.indices,self.obs,.02)
        j2,g2=m2.evaluate(self.state,self.indices,obs2,.02)
        self.assertEqual(self.model.last_checkpoint_stats, m2.last_checkpoint_stats)
        self.assertAlmostEqual(j1,j2,places=13)
        np.testing.assert_allclose(g1,g2,atol=1e-12,rtol=1e-12)

    def test_checkpoint_budgets_and_streaming_cost(self):
        jref, gref = self.model.evaluate(self.state,self.indices,self.obs,.02)
        previous_steps = None
        for budget in (1,2,4,8,12,90):
            model = Model(nx=11,ny=8,dt=.001,stride=4,cycles=3,lx=1.3,ly=.9,
                          checkpoint_budget=budget)
            j,g = model.evaluate(self.state,self.indices,self.obs,.02)
            self.assertEqual(j,jref)
            np.testing.assert_allclose(g,gref,rtol=1e-13,atol=1e-13)
            stats = model.last_checkpoint_stats
            self.assertEqual(stats['checkpoint_slots'],min(budget,11))
            self.assertEqual(stats['state_storage_bytes'],(min(budget,11)+6)*self.state.nbytes)
            if previous_steps is not None:
                self.assertLessEqual(stats['forward_steps'],previous_steps)
            previous_steps=stats['forward_steps']
            j,_=model.evaluate(self.state,self.indices,self.obs,.02,gradient=False)
            self.assertEqual(j,jref)
            self.assertEqual(model.last_checkpoint_stats['state_storage_bytes'],4*self.state.nbytes)
            self.assertEqual(model.last_checkpoint_stats['checkpoint_slots'],0)
        for budget in (0,91,1.5):
            with self.assertRaises(ValueError): Model(checkpoint_budget=budget)

    def test_control_transform_transpose_and_full_objective(self):
        rng=np.random.default_rng(200)
        transform=BackgroundTransform(self.model.shape,length=.08,lx=1.3,ly=.9)
        u,v=rng.normal(size=(2,*self.model.shape))
        self.assertAlmostEqual(np.sum(u*transform.apply(v)),np.sum(transform.transpose(u)*v),places=13)
        objective=Objective(self.model,self.state,transform,self.indices,self.obs,.02)
        v=v.ravel()*.1
        j,g=objective(v)
        d=u.ravel()/np.linalg.norm(u)
        eps=1e-5
        jp,_=objective(v+eps*d);jm,_=objective(v-eps*d)
        self.assertLess(abs((jp-jm)/(2*eps)-g@d)/max(1.,abs(g@d)),1e-7)
        self.assertTrue(np.isfinite(j))

    def test_repeated_observations_accumulate(self):
        j1,g1=self.model.evaluate(self.state,self.indices,self.obs,.02)
        j2,g2=self.model.evaluate(self.state,np.tile(self.indices,2),np.tile(self.obs,(1,2)),.02)
        self.assertAlmostEqual(j2,2*j1,places=12)
        np.testing.assert_allclose(g2,2*g1,atol=1e-12,rtol=1e-12)


if __name__=='__main__': unittest.main()
