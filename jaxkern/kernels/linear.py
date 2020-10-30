import jax.numpy as np
import objax

from jaxkern.dist import distmat
from jaxkern.kernels.base import Kernel


class Linear(Kernel):
    """
    Linear Kernel
    """

    def __init__(self, input_dim: int = 1) -> None:
        self.input_dim = input_dim

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        return distmat(
            linear_kernel,
            X,
            Y,
        )


def linear_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear kernel

    .. math:: k_i = \sum_i^N x_i-y_i

    Parameters
    ----------
    params : None
        kept for compatibility
    x : jax.numpy.ndarray
        the inputs
    y : jax.numpy.ndarray
        the inputs

    Returns
    -------
    kernel_mat : jax.numpy.ndarray
        the kernel matrix (n_samples, n_samples)

    """
    return np.sum(x * y)