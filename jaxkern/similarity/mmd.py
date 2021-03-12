from typing import Callable, Dict

import jax
import jax.numpy as np
from chex import dataclass, Array
from jaxkern.dist import sqeuclidean_distance
from jaxkern.kernels.stationary import RBF
from jaxkern.kernels.utils import covariance_matrix, gram
from jaxkern.utils import centering

jax_np = jax.numpy.ndarray


def mmd_mi(
    kernel_f: Callable, params_X: dataclass, params_Y: dataclass, X: Array, Y: Array
) -> float:
    """Maximum Mean Discrepancy

    Parameters
    ----------
    X : jax.numpy.ndarray
        array-like of shape (n_samples, n_features)
    Y : np.ndarray
        The data matrix.

    Notes
    -----

        This method is equivalent to the HSIC method.
    """
    # calculate kernel matrices
    Kx = gram(kernel_f, params_X, X, X)
    Ky = gram(kernel_f, params_Y, Y, Y)

    # center kernel matrices
    Kx = centering(Kx)
    Ky = centering(Ky)

    # get the expectrations
    A = np.mean(Kx * Ky)
    B = np.mean(np.mean(Kx, axis=0) * np.mean(Ky, axis=0))
    C = np.mean(Kx) * np.mean(Ky)

    # calculate the mmd value
    mmd_value = A - 2 * B + C

    return mmd_value


def mmd_biased(
    kernel_f: Callable,
    params_X: dataclass,
    params_Y: dataclass,
    params_XY: dataclass,
    X: Array,
    Y: Array,
) -> Array:
    """Maximum Mean Discrepancy

    Parameters
    ----------
    X : jax.numpy.ndarray
        array-like of shape (n_samples, n_features)
    Y : np.ndarray
        The data matrix.

    Notes
    -----

        This method is equivalent to the HSIC method.
    """
    # kernel matrices
    Kx = gram(kernel_f, params_X, X, X)
    Ky = gram(kernel_f, params_Y, Y, Y)
    Kxy = gram(kernel_f, params_XY, X, Y)

    return mmd_v_statistic(Kx, Ky, Kxy)


def mmd_unbiased(
    kernel_f: Callable,
    params_X: dataclass,
    params_Y: dataclass,
    params_XY: dataclass,
    X: Array,
    Y: Array,
) -> Array:
    """Maximum Mean Discrepancy

    Parameters
    ----------
    X : jax.numpy.ndarray
        array-like of shape (n_samples, n_features)
    Y : np.ndarray
        The data matrix.

    Notes
    -----

        This method is equivalent to the HSIC method.
    """
    n_samples, m_samples = X.shape[0], Y.shape[0]

    # constants
    a00 = 1.0 / (n_samples * (n_samples - 1.0))
    a11 = 1.0 / (m_samples * (m_samples - 1.0))
    a01 = -1.0 / (n_samples * m_samples)

    # kernel matrices
    Kx = gram(kernel_f, params_X, X, X)
    Ky = gram(kernel_f, params_Y, Y, Y)
    Kxy = gram(kernel_f, params_XY, X, Y)

    return (
        2 * a01 * np.mean(Kxy)
        + a00 * (np.sum(Kx) - n_samples)
        + a11 * (np.sum(Ky) - m_samples)
    )


def mmd_u_statistic(K_x, K_y, K_xy):
    """
    Calculate the MMD unbiased u-statistic
    """

    K_x = K_x - np.diag(np.diag(K_x))
    K_y = K_y - np.diag(np.diag(K_y))
    n_samples, m_samples = K_x.shape[0], K_y.shape[0]

    # Term 1
    A = np.sum(K_x) / (np.power(n_samples, 2) - n_samples)

    # Term 2
    B = np.sum(K_y) / (np.power(m_samples, 2) - m_samples)

    # Term 3
    C = np.mean(K_xy)

    return A + B - 2 * C


def mmd_v_statistic(K_x, K_y, K_xy) -> np.ndarray:
    """
    Calculate the MMD biased v-statistic
    """

    # Term 1
    A = np.mean(K_x)

    # Term 2
    B = np.mean(K_y)

    # Term 3
    C = np.mean(K_xy)

    return A + B - 2 * C
