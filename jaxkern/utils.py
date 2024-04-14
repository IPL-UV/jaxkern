import jax
import jax.numpy as np
from objax.typing import JaxArray

_float_eps = np.finfo("float").eps


def ensure_min_eps(x: JaxArray) -> JaxArray:
    """Ensures no overflow or round-off errors"""
    return np.maximum(_float_eps, x)


def centering(kernel_mat: JaxArray) -> JaxArray:
    """Calculates the centering matrix for the kernel
    Particularly useful in unsupervised kernel methods like
    HSIC and MMD.

    Parameters
    ----------
    kernel_mat : JaxArray
        PSD kernel matrix, (n_samples, n_samples)

    Returns
    -------
    centered_kernel_mat : JaxArray
        centered PSD kernel matrix, (n_samples, n_samples)
    """
    n_samples = np.shape(kernel_mat)[0]

    identity = np.eye(n_samples)

    H = identity - (1.0 / n_samples) * np.ones((n_samples, n_samples))

    kernel_mat = np.einsum("ij,jk,kl->il", H, kernel_mat, H)

    return kernel_mat
