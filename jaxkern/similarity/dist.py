from jaxkern.utils import centering
from typing import Callable
import jax
import jax.numpy as np
import objax
from jaxkern.similarity.hsic import nhsic_cka
from jaxkern.dist import distmat, sqeuclidean_distance
from jaxkern.kernels.linear import Linear, linear_kernel
from jaxkern.kernels.stationary import RBF
from jaxkern.kernels.utils import kernel_matrix


def distance_corr(X: jax.numpy.ndarray, sigma=1.0) -> float:
    """Distance correlation"""
    X = distmat(sqeuclidean_distance, X, X)
    X = np.exp(-X / (2 * sigma ** 2))
    return np.mean(X)


def energy_distance(X: np.ndarray, Y: np.ndarray, sigma=1.0) -> float:
    """Distance correlation"""
    n_samples, m_samples = X.shape[0], Y.shape[0]
    a00 = -1.0 / (n_samples * n_samples)
    a11 = -1.0 / (m_samples * m_samples)
    a01 = 1.0 / (n_samples * m_samples)
    # X = distmat(sqeuclidean_distance, X, X)
    # X = np.exp(-X / (2 * sigma ** 2))

    # calculate distances
    dist_x = sqeuclidean_distance(X, X)
    dist_y = sqeuclidean_distance(Y, Y)
    dist_xy = sqeuclidean_distance(X, Y)

    return 2 * a01 * dist_xy + a00 * dist_x + a11 * dist_y