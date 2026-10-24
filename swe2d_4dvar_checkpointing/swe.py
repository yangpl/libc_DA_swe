"""NumPy interface to the C shallow-water model and checkpointed adjoint."""
from pathlib import Path
import ctypes as ct
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent
ERRORS = {1: 'invalid model arguments', 2: 'allocation failed',
          3: 'nonfinite or dry state', 4: 'two-dimensional CFL exceeds 0.9',
          5: 'nonfinite cost/gradient or checkpoint invariant failure'}


class CheckpointStats(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in (
        'checkpoint_slots', 'state_storage_bytes', 'forward_steps',
        'recomputed_steps', 'stage_recomputations', 'reverse_steps',
        'checkpoint_saves', 'checkpoint_restores')]

    def as_dict(self):
        return {name: int(getattr(self, name)) for name, _ in self._fields_}


def build_library():
    subprocess.run(['make', '--no-print-directory', '-s', 'libswe.so'],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return ROOT / 'libswe.so'


class ModelError(RuntimeError):
    pass


class Model:
    def __init__(self, nx=60, ny=60, dt=.001, stride=20, cycles=15, lx=1., ly=1.,
                 checkpoint_budget=8):
        if (not isinstance(nx, int) or not isinstance(ny, int) or
                not isinstance(stride, int) or not isinstance(cycles, int) or
                min(nx, ny) < 2 or max(nx, ny) > 4096 or min(stride, cycles) < 1 or
                not np.all(np.isfinite([dt, lx, ly])) or min(dt, lx, ly) <= 0):
            raise ValueError('Invalid grid, time step, or observation schedule')
        if stride * cycles > 1000000:
            raise ValueError('Total steps must be <= 1000000')
        if not isinstance(checkpoint_budget, int) or not 1 <= checkpoint_budget <= 90:
            raise ValueError('Checkpoint budget must be an integer from 1 to 90')
        self.checkpoint_budget = checkpoint_budget
        self.last_checkpoint_stats = None
        self.nx, self.ny, self.dt = nx, ny, dt
        self.stride, self.cycles, self.lx, self.ly = stride, cycles, lx, ly
        self.shape = (3, ny, nx)
        self.lib = ct.CDLL(str(build_library()))
        self.array = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        ints = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        base = [self.array, ct.c_int, ct.c_int, ct.c_double, ct.c_double,
                ct.c_double, ct.c_int, ct.c_int]
        self.lib.swe_forward.argtypes = base + [self.array, self.array]
        self.lib.swe_forward.restype = ct.c_int
        self.lib.swe_evaluate.argtypes = base + [ct.c_int, ints, self.array, ct.c_double,
                                                self.array, ct.c_void_p, self.array]
        self.lib.swe_evaluate.restype = ct.c_int
        self.lib.swe_evaluate_binomial.argtypes = base + [ct.c_int, ints, self.array,
            ct.c_double, ct.c_int, self.array, ct.c_void_p, self.array,
            ct.POINTER(CheckpointStats)]
        self.lib.swe_evaluate_binomial.restype = ct.c_int

    def _args(self, state):
        state = np.ascontiguousarray(state, dtype=np.float64)
        if state.shape != self.shape:
            raise ValueError(f'Expected state shape {self.shape}, got {state.shape}')
        return [state, self.nx, self.ny, self.lx, self.ly, self.dt,
                self.stride * self.cycles, self.stride]

    @staticmethod
    def _check(code):
        if code:
            raise ModelError(ERRORS.get(code, f'C model error {code}'))

    def forward(self, initial):
        trajectory = np.empty((self.cycles + 1, *self.shape))
        diagnostics = np.empty(2)
        self._check(self.lib.swe_forward(*self._args(initial), trajectory, diagnostics))
        return trajectory, dict(min_depth=float(diagnostics[0]), max_cfl=float(diagnostics[1]))

    def evaluate(self, initial, indices, observations, obs_std, gradient=True):
        # Validate before casting: fractional or overflowing indices must not
        # silently become valid sensors when converted to C integers.
        indices = np.asarray(indices)
        if (indices.ndim != 1 or not indices.size or indices.dtype.kind not in 'iu'
                or np.any(indices < 0) or np.any(indices >= self.nx * self.ny)):
            raise ValueError('Sensor indices must be a nonempty vector of in-range integers')
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        observations = np.ascontiguousarray(observations, dtype=np.float64)
        if indices.ndim != 1 or observations.shape != (self.cycles, len(indices)):
            raise ValueError('Observation shape must be (cycles, number of sensors)')
        if not np.isfinite(obs_std) or obs_std <= 0 or not np.all(np.isfinite(observations)):
            raise ValueError('Observations must be finite and obs_std must be finite and positive')
        cost, diagnostics = np.empty(1), np.empty(2)
        grad = np.empty(self.shape) if gradient else None
        stats = CheckpointStats()
        code = self.lib.swe_evaluate_binomial(*self._args(initial), len(indices), indices,
                    observations, obs_std, self.checkpoint_budget, cost,
                    None if grad is None else grad.ctypes.data, diagnostics, ct.byref(stats))
        self.last_checkpoint_stats = stats.as_dict()
        self._check(code)
        if not np.isfinite(cost[0]) or (grad is not None and not np.all(np.isfinite(grad))):
            raise ModelError('Nonfinite observation cost or gradient')
        return float(cost[0]), grad
