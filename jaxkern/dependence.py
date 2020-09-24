from typing import Callable, Dict
import jax
import jax.numpy as np

from jaxkern.kernels import gram, covariance_matrix
from jaxkern.utils import centering

jax_np = jax.numpy.ndarray


def hsic(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
    bias: bool = False,
) -> float:
    """Normalized HSIC (Tangent Kernel Alignment)

    A normalized variant of HSIC method which divides by
    the HS-Norm of each dataset.

    Parameters
    ----------
    X : jax.numpy.ndarray
        the input value for one dataset

    Y : jax.numpy.ndarray
        the input value for the second dataset

    kernel : Callable
        the kernel function to be used for each of the kernel
        calculations

    params_x : Dict[str, float]
        a dictionary of parameters to be used for calculating the
        kernel function for X

    params_y : Dict[str, float]
        a dictionary of parameters to be used for calculating the
        kernel function for Y

    Returns
    -------
    cka_value : float
        the normalized hsic value.

    Notes
    -----

        This is a metric that is similar to the correlation, [0,1]
    """
    # kernel matrix
    Kx = covariance_matrix(kernel, params_x, X, X)
    Ky = covariance_matrix(kernel, params_y, Y, Y)
    # print(Kx.min(), Kx.max(), Ky.min(), Ky.max())
    # # print(Kx.min(), Kx.max(), Ky.min(), Ky.max())

    # import matplotlib.pyplot as plt

    # plt.imshow(Kx)
    # plt.colorbar()
    # plt.show()

    # # center kernel matrices
    # n_samples = Kx.shape[0]
    # H = np.eye(n_samples) - (1.0 / n_samples) * np.ones((n_samples, n_samples))
    # Kx_ = np.dot(Kx, H)
    # Ky_ = np.dot(Ky, H)
    # print(Kx_.min(), Kx_.max(), Ky_.min(), Ky_.max())

    Kx = centering(Kx)
    Ky = centering(Ky)
    # print(Kx.min(), Kx.max(), Ky.min(), Ky.max())

    # import matplotlib.pyplot as plt

    # plt.imshow(Kx)
    # plt.colorbar()
    # plt.show()
    #
    # K = np.dot(Kx, Ky.T)
    # print(K.min(), K.max())

    # return np.mean(K)
    hsic_value = np.sum(Kx * Ky)
    if bias:
        bias = 1 / (Kx.shape[0] ** 2)
    else:
        bias = 1 / (Kx.shape[0] - 1) ** 2
    return bias * hsic_value


def nhsic_cka(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
) -> float:
    """Normalized HSIC (Tangent Kernel Alignment)

    A normalized variant of HSIC method which divides by
    the HS-Norm of each dataset.

    Parameters
    ----------
    X : jax.numpy.ndarray
        the input value for one dataset

    Y : jax.numpy.ndarray
        the input value for the second dataset

    kernel : Callable
        the kernel function to be used for each of the kernel
        calculations

    params_x : Dict[str, float]
        a dictionary of parameters to be used for calculating the
        kernel function for X

    params_y : Dict[str, float]
        a dictionary of parameters to be used for calculating the
        kernel function for Y

    Returns
    -------
    cka_value : float
        the normalized hsic value.

    Notes
    -----

        This is a metric that is similar to the correlation, [0,1]

    References
    ----------
    """
    # calculate hsic normally (numerator)
    # Pxy = hsic(X, Y, kernel, params_x, params_y)

    # # calculate denominator (normalize)
    # Px = np.sqrt(hsic(X, X, kernel, params_x, params_x))
    # Py = np.sqrt(hsic(Y, Y, kernel, params_y, params_y))

    # # print(Pxy, Px, Py)

    # # kernel tangent alignment value (normalized hsic)
    # cka_value = Pxy / (Px * Py)
    Kx = covariance_matrix(kernel, params_x, X, X)
    Ky = covariance_matrix(kernel, params_y, Y, Y)

    Kx = centering(Kx)
    Ky = centering(Ky)

    cka_value = np.sum(Kx * Ky) / np.linalg.norm(Kx) / np.linalg.norm(Ky)

    return cka_value


def nhsic_ka(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
) -> float:

    Kx = covariance_matrix(kernel, params_x, X, X)
    Ky = covariance_matrix(kernel, params_y, Y, Y)

    cka_value = np.sum(Kx * Ky) / np.linalg.norm(Kx) / np.linalg.norm(Ky)

    return cka_value


def nhsic_cca(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
    epsilon: float = 1e-5,
    bias: bool = False,
) -> float:
    """Normalized HSIC (Tangent Kernel Alignment)

    A normalized variant of HSIC method which divides by
    the HS-Norm of each dataset.

    Parameters
    ----------
    X : jax.numpy.ndarray
        the input value for one dataset

    Y : jax.numpy.ndarray
        the input value for the second dataset

    kernel : Callable
        the kernel function to be used for each of the kernel
        calculations

    params_x : Dict[str, float]
        a dictionary of parameters to be used for calculating the
        kernel function for X

    params_y : Dict[str, float]
        a dictionary of parameters to be used for calculating the
        kernel function for Y

    Returns
    -------
    cka_value : float
        the normalized hsic value.

    Notes
    -----

        This is a metric that is similar to the correlation, [0,1]
    """
    n_samples = X.shape[0]

    # kernel matrix
    Kx = gram(kernel, params_x, X, X)
    Ky = gram(kernel, params_y, Y, Y)

    # center kernel matrices
    Kx = centering(Kx)
    Ky = centering(Ky)

    K_id = np.eye(Kx.shape[0])
    Kx_inv = np.linalg.inv(Kx + epsilon * n_samples * K_id)
    Ky_inv = np.linalg.inv(Ky + epsilon * n_samples * K_id)

    Rx = np.dot(Kx, Kx_inv)
    Ry = np.dot(Ky, Ky_inv)

    hsic_value = np.sum(Rx * Ry)

    if bias:
        bias = 1 / (Kx.shape[0] ** 2)
    else:
        bias = 1 / (Kx.shape[0] - 1) ** 2
    return bias * hsic_value


def _hsic_uncentered(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
) -> float:
    """A method to calculate the uncentered HSIC version"""
    # kernel matrix
    Kx = gram(kernel, params_x, X, X)
    Ky = gram(kernel, params_y, Y, Y)

    #
    K = np.dot(Kx, Ky.T)

    hsic_value = np.mean(K)

    return hsic_value


def mmd_mi(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
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
    Kx = gram(kernel, params_x, X, X)
    Ky = gram(kernel, params_y, Y, Y)

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


def mmd_centered(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
    params_xy: Dict[str, float],
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
    Kx = gram(kernel, params_x, X, X)
    Ky = gram(kernel, params_y, Y, Y)
    Kxy = gram(kernel, params_xy, X, Y)

    # center kernel matrices
    Kx = centering(Kx)
    Ky = centering(Ky)
    Kxy = centering(Kxy)

    #
    K = np.dot(Kx, Ky.T)

    hsic_value = np.mean(K)

    return hsic_value