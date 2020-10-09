from jaxkern.dist import sqeuclidean_distance
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

    Kx = centering(Kx)
    Ky = centering(Ky)

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


def nhsic_nbs(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
) -> float:
    """Normalized Bures Similarity (NBS)

    A normalized variant of HSIC method which divides by
    the HS-Norm of the eigenvalues of each dataset.

    ..math::
        \\rho(K_x, K_y) = \\
        \\text{Tr} ( K_x^{1/2} K_y K_x^{1/2)})^{1/2} \\
        \ \\text{Tr} (K_x) \\text{Tr} (K_y)

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

    @article{JMLR:v18:16-296,
    author  = {Austin J. Brockmeier and Tingting Mu and Sophia Ananiadou and John Y. Goulermas},
    title   = {Quantifying the Informativeness of Similarity Measurements},
    journal = {Journal of Machine Learning Research},
    year    = {2017},
    volume  = {18},
    number  = {76},
    pages   = {1-61},
    url     = {http://jmlr.org/papers/v18/16-296.html}
    }
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

    # numerator
    numerator = np.real(np.linalg.eigvals(np.dot(Kx, Ky)))

    # clip rogue numbers
    numerator = np.sqrt(np.clip(numerator, 0.0))

    numerator = np.sum(numerator)

    # denominator
    denominator = np.sqrt(np.trace(Kx) * np.trace(Ky))

    # return nbs value
    return numerator / denominator


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


def mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable,
    params_x: Dict[str, float],
    params_y: Dict[str, float],
    params_xy: Dict[str, float],
    bias: bool = False,
    center: bool = False,
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
    n_samples, m_samples = X.shape[0], Y.shape[0]

    # constants
    a00 = 1.0 / (n_samples * (n_samples - 1.0))
    a11 = 1.0 / (m_samples * (m_samples - 1.0))
    a01 = -1.0 / (n_samples * m_samples)

    # kernel matrices
    Kx = gram(kernel, params_x, X, X)
    Ky = gram(kernel, params_y, Y, Y)
    Kxy = gram(kernel, params_xy, X, Y)

    if bias:
        mmd = np.mean(Kx) + np.mean(Ky) - 2 * np.mean(Kxy)
        return np.where(mmd >= 0.0, np.sqrt(mmd), 0.0)
    else:
        return (
            2 * a01 * np.mean(Kxy)
            + a00 * (np.sum(Kx) - n_samples)
            + a11 * (np.sum(Ky) - m_samples)
        )


