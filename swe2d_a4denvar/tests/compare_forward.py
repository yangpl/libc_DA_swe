#!/usr/bin/env python3
"""Optional comparison with the current hand-coded 4D-Var forward model.

Compiles the sibling C source in a temporary directory. No sibling files change.
Not required for standalone tests or assimilation.
"""
import ctypes as ct
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swe import Model


def main():
    root = Path(__file__).resolve().parents[2]
    source = root/'swe2d_4dvar_checkpointing/swe2d_4dvar.c'
    rng = np.random.default_rng(20261111)
    with tempfile.TemporaryDirectory() as directory:
        libpath = Path(directory)/'reference.so'
        subprocess.run(['gcc', '-O3', '-std=c11', '-fPIC', '-shared', str(source), '-lm', '-o', str(libpath)], check=True)
        lib = ct.CDLL(str(libpath))
        for nx, ny, stride, cycles in ((2, 2, 1, 1), (9, 7, 5, 7), (60, 60, 20, 25)):
            model = Model(nx=nx, ny=ny, stride=stride, cycles=cycles)
            state = rng.normal(0, .02, model.shape); state[0] += 1.1
            actual, diagnostics = model.forward(state)
            reference, diag = np.empty_like(actual), np.empty(2)
            lib.swe_forward.argtypes = model.lib.swe_forward.argtypes
            lib.swe_forward.restype = ct.c_int
            code = lib.swe_forward(state, nx, ny, 1., 1., .001, stride*cycles, stride, reference, diag)
            assert code == 0
            np.testing.assert_array_equal(actual, reference)
            np.testing.assert_array_equal([diagnostics['min_depth'], diagnostics['max_cfl']], diag)
            print(f'{nx}x{ny}, {stride*cycles} steps: forward trajectory and diagnostics identical')


if __name__ == '__main__':
    main()
