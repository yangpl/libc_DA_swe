"""Fixed-ensemble strong-constraint 4D-EnVar; no tangent-linear or adjoint model.

X and Y contain centred anomalies divided by sqrt(Ne-1).
The innovation uses G(background), not the mean of G(ensemble).
One coefficient vector controls the entire observation window.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
from swe import ModelError


def make_ensemble(background, transform, size=40, seed=20261019, spread=1.):
    background = np.asarray(background, dtype=float)
    if (not isinstance(size, int) or isinstance(size, bool) or size < 2
            or not np.isfinite(spread) or spread <= 0
            or background.shape != transform.shape or not np.all(np.isfinite(background))):
        raise ValueError('Invalid ensemble size, spread, or background')
    rng = np.random.default_rng(seed)
    perturbations = np.array([transform.apply(rng.normal(size=background.shape)) for _ in range(size)])
    perturbations -= perturbations.mean(axis=0)
    return background + spread*perturbations


class QuadraticObjective:
    """J(w)=0.5*w.T*w+0.5*||(Y*w-d)/sigma||^2, with diagonal R."""
    def __init__(self, anomalies, innovation, obs_std):
        y = np.asarray(anomalies, dtype=float)
        d = np.asarray(innovation, dtype=float)
        if y.ndim != 2 or min(y.shape) < 1 or d.shape != (y.shape[0],):
            raise ValueError('Expected anomalies (nobservations, ncoefficients) and innovation (nobservations,)')
        sigma = np.broadcast_to(np.asarray(obs_std, dtype=float), d.shape)
        if not all(np.all(np.isfinite(a)) for a in (y, d, sigma)) or np.any(sigma <= 0):
            raise ValueError('Anomalies and innovations must be finite; observation standard deviations must be positive')
        self.y = y / sigma[:, None]
        self.d = d / sigma
        if not np.all(np.isfinite(self.y)) or not np.all(np.isfinite(self.d)):
            raise ValueError('Whitened observations overflow')
        self.size = y.shape[1]

    def __call__(self, w):
        w = np.asarray(w, dtype=float)
        if w.shape != (self.size,) or not np.all(np.isfinite(w)):
            raise ValueError('Invalid ensemble coefficients')
        residual = self.y @ w - self.d
        return float(.5*(w @ w + residual @ residual)), w + self.y.T @ residual

    def hessian_product(self, p):
        return p + self.y.T @ (self.y @ p)

    def solve(self, method='cholesky', rtol=1e-12, maxiter=None):
        if method not in ('cholesky', 'cg') or not np.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError('Invalid solver or tolerance')
        if maxiter is not None and (not isinstance(maxiter, int) or maxiter < 1):
            raise ValueError('maxiter must be a positive integer')
        history = []

        def record(w):
            cost, gradient = self(w)
            jb = float(.5*(w @ w))
            history.append(dict(iteration=len(history), cost=cost, background_cost=jb,
                                observation_cost=cost-jb,
                                gradient_inf=float(np.linalg.norm(gradient, np.inf))))

        record(np.zeros(self.size))
        rhs = self.y.T @ self.d
        if method == 'cholesky':
            a = np.eye(self.size) + self.y.T @ self.y
            w = cho_solve(cho_factor(a, lower=True), rhs)
            record(w)
        else:
            operator = LinearOperator((self.size, self.size), matvec=self.hessian_product, dtype=float)
            w, info = cg(operator, rhs, rtol=rtol, atol=0.,
                         maxiter=maxiter or 10*self.size, callback=record)
            if info != 0:
                raise RuntimeError(f'Ensemble CG did not converge (info={info})')
            if len(history) == 1:
                record(w)
        residual = self(w)[1]
        relative_residual = float(np.linalg.norm(residual)/max(1., np.linalg.norm(rhs)))
        if not np.all(np.isfinite(w)) or relative_residual > max(1e-10, 10*rtol):
            raise RuntimeError(f'Ensemble solve residual too large: {relative_residual}')
        return w, dict(method=method, success=True, iterations=len(history)-1,
                       relative_residual=relative_residual, history=history)


class EnsembleProblem:
    """Build X0 and stacked Y using sequential forward integrations.

    Retains initial anomalies and member observations, not all member trajectories.
    Invalid members fail explicitly; no clipping or selective resampling is used.
    """
    def __init__(self, model, background, members, observations, obs_std,
                 *, indices=None, receivers=None):
        self.background = np.array(background, dtype=float, copy=True)
        members = np.asarray(members, dtype=float)
        if (self.background.shape != model.shape or members.ndim != 4
                or members.shape[1:] != model.shape or len(members) < 2
                or not np.all(np.isfinite(members)) or not np.all(np.isfinite(self.background))):
            raise ValueError('Invalid background or ensemble shape/values')
        if not np.allclose(members.mean(axis=0), self.background, rtol=1e-12, atol=1e-14):
            raise ValueError('The initial ensemble mean must equal the background')
        self.size = len(members)
        scale = np.sqrt(self.size-1)
        self.x = (members-members.mean(axis=0)).reshape(self.size, -1).T / scale
        trajectory, diag = model.forward(self.background)
        self.background_observations = model.observe(trajectory, indices=indices, receivers=receivers)
        observations = np.asarray(observations, dtype=float)
        if observations.shape != self.background_observations.shape or not np.all(np.isfinite(observations)):
            raise ValueError('Observations must be finite with shape (cycles, nsensors)')
        sigma = np.broadcast_to(np.asarray(obs_std, dtype=float), observations.shape)
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError('Observation standard deviations must be finite and positive')
        self.member_observations = np.empty((self.size, *observations.shape))
        self.diagnostics = dict(diag)
        for k, member in enumerate(members):
            try:
                trajectory, diag = model.forward(member)
            except ModelError as exc:
                raise ModelError(f'Ensemble member {k}: {exc}') from exc
            self.member_observations[k] = model.observe(trajectory, indices=indices, receivers=receivers)
            self.diagnostics['min_depth'] = min(self.diagnostics['min_depth'], diag['min_depth'])
            self.diagnostics['max_cfl'] = max(self.diagnostics['max_cfl'], diag['max_cfl'])
        self.ensemble_mean_observations = self.member_observations.mean(axis=0)
        self.y = ((self.member_observations-self.ensemble_mean_observations)
                  .reshape(self.size, -1).T / scale)
        self.innovation = (observations-self.background_observations).ravel()
        self.objective = QuadraticObjective(self.y, self.innovation, sigma.ravel())

    def initial_state(self, coefficients):
        coefficients = np.asarray(coefficients, dtype=float)
        if coefficients.shape != (self.size,) or not np.all(np.isfinite(coefficients)):
            raise ValueError('Invalid ensemble coefficients')
        return self.background + (self.x @ coefficients).reshape(self.background.shape)

    def predicted_observations(self, coefficients):
        self.initial_state(coefficients)  # Validate coefficient shape and values.
        return self.background_observations + (self.y @ coefficients).reshape(self.background_observations.shape)
