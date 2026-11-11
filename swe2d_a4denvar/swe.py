"""NumPy interface to the standalone, forward-only C shallow-water model."""
import ctypes as ct
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
ERRORS = {1: 'invalid model arguments', 2: 'allocation failed',
          3: 'nonfinite or dry state', 4: 'two-dimensional CFL exceeds 0.9',
          5: 'nonfinite sampled value'}


class ModelError(RuntimeError):
    pass


def build_library():
    subprocess.run(['make', '--no-print-directory', '-s', 'libswe.so'],
                   cwd=ROOT, check=True)
    return ROOT / 'libswe.so'


class Model:
    def __init__(self, nx=60, ny=60, dt=.001, stride=20, cycles=15, lx=1., ly=1.):
        if (any(not isinstance(v, int) or isinstance(v, bool) for v in (nx, ny, stride, cycles))
                or min(nx, ny) < 2 or max(nx, ny) > 4096 or min(stride, cycles) < 1
                or stride*cycles > 1000000 or not np.all(np.isfinite([dt, lx, ly]))
                or min(dt, lx, ly) <= 0):
            raise ValueError('Invalid grid, time step, or observation schedule')
        self.nx, self.ny, self.dt = nx, ny, dt
        self.stride, self.cycles, self.lx, self.ly = stride, cycles, lx, ly
        self.shape = (3, ny, nx)
        self.lib = ct.CDLL(str(build_library()))
        array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        self.lib.swe_forward.argtypes = [array, ct.c_int, ct.c_int, ct.c_double,
            ct.c_double, ct.c_double, ct.c_int, ct.c_int, array, array]
        self.lib.swe_forward.restype = ct.c_int
        self.lib.swe_sample_field.argtypes = [array, ct.c_int, ct.c_int,
            ct.c_double, ct.c_double, ct.c_int, array, array, array]
        self.lib.swe_sample_field.restype = ct.c_int

    @staticmethod
    def _check(code):
        if code:
            raise ModelError(ERRORS.get(code, f'C model error {code}'))

    def forward(self, initial):
        initial = np.ascontiguousarray(initial, dtype=np.float64)
        if initial.shape != self.shape:
            raise ValueError(f'Expected state shape {self.shape}')
        trajectory = np.empty((self.cycles+1, *self.shape))
        diag = np.empty(2)
        self._check(self.lib.swe_forward(initial, self.nx, self.ny, self.lx,
                    self.ly, self.dt, self.stride*self.cycles, self.stride, trajectory, diag))
        return trajectory, dict(min_depth=float(diag[0]), max_cfl=float(diag[1]))

    def sample_field(self, values, receivers):
        receivers = np.asarray(receivers, dtype=np.float64)
        if (receivers.ndim != 2 or receivers.shape[1] != 2 or not len(receivers)
                or not np.all(np.isfinite(receivers)) or np.any(receivers < 0)
                or np.any(receivers > [self.lx, self.ly])):
            raise ValueError('Receivers must be finite (nobs, 2) coordinates inside the domain')
        values = np.ascontiguousarray(values, dtype=np.float64)
        if values.shape != (self.ny, self.nx):
            raise ValueError('Incorrect scalar field shape')
        samples = np.empty(len(receivers))
        self._check(self.lib.swe_sample_field(values, self.nx, self.ny, self.lx,
                    self.ly, len(receivers), np.ascontiguousarray(receivers[:, 0]),
                    np.ascontiguousarray(receivers[:, 1]), samples))
        return samples

    def observe(self, trajectory, *, indices=None, receivers=None):
        """Sample depth at observation times, excluding t=0; choose one location API."""
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.shape != (self.cycles+1, *self.shape):
            raise ValueError('Incorrect trajectory shape')
        if (indices is None) == (receivers is None):
            raise ValueError('Supply exactly one of indices and receivers')
        if receivers is not None:
            samples = np.array([self.sample_field(q[0], receivers) for q in trajectory[1:]])
        else:
            indices = np.asarray(indices)
            if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in 'iu'
                    or np.any(indices < 0) or np.any(indices >= self.nx*self.ny)):
                raise ValueError('Sensor indices must be a nonempty vector of in-range integers')
            samples = trajectory[1:, 0].reshape(self.cycles, -1)[:, indices]
        if not np.all(np.isfinite(samples)):
            raise ModelError('Nonfinite sampled value')
        return samples
