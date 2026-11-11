"""DCT background covariance square root reused from the 4D-Var experiment."""
import numpy as np
from scipy.fft import dctn, idctn

class BackgroundTransform:
    """x = xb + L v, B = L L^T. Symmetric DCT Gaussian square root.

    Sigma is the spatially averaged pointwise standard deviation, not the
    standard deviation of the DCT coefficients. Reflecting covariance edges
    need not match the model's open physical boundary condition.
    """
    def __init__(self, shape, sigma=(.03, .015, .015), length=.06, lx=1., ly=1.):
        if (len(shape) != 3 or shape[0] != 3 or min(shape[1:]) < 2
                or np.asarray(sigma).shape != (3,) or not np.all(np.isfinite(sigma))
                or min(sigma) <= 0 or not np.all(np.isfinite([length, lx, ly]))
                or length < 0 or min(lx, ly) <= 0):
            raise ValueError('Invalid covariance shape, scales, or domain')
        _, ny, nx = shape
        ky, kx = np.meshgrid(np.arange(ny)*np.pi/ly, np.arange(nx)*np.pi/lx, indexing='ij')
        kernel = np.exp(-.5 * length**2 * (kx*kx + ky*ky))
        kernel *= np.sqrt(nx*ny / np.sum(kernel**2))
        self.weights = np.array(sigma)[:, None, None] * kernel
        self.shape = shape

    def apply(self, v):
        return idctn(self.weights * dctn(np.asarray(v).reshape(self.shape),
                                       axes=(-2, -1), norm='ortho'),
                     axes=(-2, -1), norm='ortho')

    transpose = apply


