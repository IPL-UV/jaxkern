from typing import Callable

import jax.numpy as np

from jaxkern.dist import sqeuclidean_distance
from jaxkern.kernels.stationary import RBF
from jaxkern.kernels.utils import kernel_matrix
from chex import Array


def distance_corr(X: Array, Y: Array) -> float:
    """Distance correlation"""
    a = kernel_matrix(sqeuclidean_distance, X, X)
    b = kernel_matrix(sqeuclidean_distance, Y, Y)
    n_samples = X.shape[0]

    A = (
        a
        - np.expand_dims(np.mean(a, axis=0), axis=0)
        - np.expand_dims(np.mean(a, axis=1), axis=1)
        + np.mean(a)
    )
    B = (
        b
        - np.expand_dims(np.mean(b, axis=0), axis=0)
        - np.expand_dims(np.mean(b, axis=1), axis=1)
        + np.mean(b)
    )

    # calculate different terms
    term1 = np.sum(A ** 2) / n_samples ** 2
    term2 = np.sum(B ** 2) / n_samples ** 2
    term3 = np.sum(A * B) / n_samples ** 2

    # calculate correlation
    return np.sqrt(term3) / np.sqrt(np.sqrt(term1) * np.sqrt(term2))


def energy_distance(
    X: Array,
    Y: Array,
) -> float:
    """Distance correlation"""
    n_samples, m_samples = X.shape[0], Y.shape[0]
    a00 = -1.0 / (n_samples * n_samples)
    a11 = -1.0 / (m_samples * m_samples)
    a01 = 1.0 / (n_samples * m_samples)

    # calculate distances
    dist_xy = sqeuclidean_distance(X, Y)
    dist_x = sqeuclidean_distance(X, X)
    dist_y = sqeuclidean_distance(Y, Y)

    return 2 * a01 * dist_xy + a00 * dist_x + a11 * dist_y