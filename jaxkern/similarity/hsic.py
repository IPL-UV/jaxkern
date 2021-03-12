from jaxkern.kernels.kernels import covariance_matrix
import math
from typing import Callable, Optional

import jax
import jax.numpy as np
from chex import dataclass, Array
from jaxkern.kernels.utils import centering_matrix


def hsic_biased(
    kernel_f: Callable, params_X: dataclass, params_Y: dataclass, X: Array, Y: Array
) -> Array:

    # calculate the kernel matrices
    K_x = covariance_matrix(params=params_X, func=kernel_f, x=X, y=X)
    K_y = covariance_matrix(params=params_Y, func=kernel_f, x=Y, y=Y)

    # calculate hsic
    hsic = hsic_v_statistic_trace(K_x, K_y)

    return hsic


def hsic_u_statistic_einsum(K_x: np.ndarray, K_y: np.ndarray) -> np.ndarray:
    """
    Calculate the unbiased statistic
    """
    n_samples = K_x.shape[0]

    # Term 1
    a = math.factorial(n_samples - 2) / math.factorial(n_samples)
    A = np.einsum("ij,ij->", K_x, K_y)

    # Term 2
    b = math.factorial(n_samples - 4) / math.factorial(n_samples)
    B = np.einsum("ij,kl->", K_x, K_y)

    # Term 3
    c = 2 * math.factorial(n_samples - 3) / math.factorial(n_samples)
    C = np.einsum("ij,ik->", K_x, K_y)
    return a * A.squeeze() + b * B.squeeze() - c * C.squeeze()


def hsic_u_statistic_dot(K_x: Array, K_y: Array) -> Array:
    """
    Calculate the unbiased statistic
    """
    n_samples = K_x.shape[0]

    K_xd = K_x - np.diag(np.diag(K_x))
    K_yd = K_y - np.diag(np.diag(K_y))
    K_xy = K_xd @ K_yd.T

    # Term 1
    a = 1 / n_samples / (n_samples - 3)
    A = np.trace(K_xy)

    # Term 2
    ones_v = np.ones((n_samples, 1))
    b = a / (n_samples - 1) / (n_samples - 2)
    B = ones_v.T @ K_xd @ ones_v @ ones_v.T @ K_yd @ ones_v

    # Term 3
    c = (a * 2) / (n_samples - 2)
    C = ones_v.T @ K_xy @ ones_v
    return a * A.squeeze() + b * B.squeeze() - c * C.squeeze()


def hsic_v_statistic_einsum(K_x, K_y):
    n_samples = K_x.shape[0]
    A = np.einsum("ij,ij->", K_x, K_y) / n_samples ** 2
    B = np.einsum("ij,kl->", K_x, K_y) / n_samples ** 4
    C = np.einsum("ij,ik->", K_x, K_y) / n_samples ** 3
    return A + B - 2 * C


def hsic_v_statistic_trace(K_x, K_y):
    # get the samples
    n_samples = K_x.shape[0]

    # get the centering matrix
    H = centering_matrix(n_samples)

    # calculat HSIC
    return np.einsum("ij,ji->", K_x @ H, K_y @ H) / n_samples ** 2


def hsic_v_statistic_rff(Z_x, Z_y):
    n_samples = Z_x.shape[0]
    Z_x = Z_x - np.mean(Z_x, axis=0)
    Z_y = Z_y - np.mean(Z_y, axis=0)
    featCov = np.dot(Z_x.T, Z_y) / n_samples
    return np.linalg.norm(featCov) ** 2


# def hsic(
#     X: np.ndarray,
#     Y: np.ndarray,
#     kernel: Callable,
#     params_x: Dict[str, float],
#     params_y: Dict[str, float],
#     bias: bool = False,
# ) -> float:
#     """Normalized HSIC (Tangent Kernel Alignment)

#     A normalized variant of HSIC method which divides by
#     the HS-Norm of each dataset.

#     Parameters
#     ----------
#     X : jax.numpy.ndarray
#         the input value for one dataset

#     Y : jax.numpy.ndarray
#         the input value for the second dataset

#     kernel : Callable
#         the kernel function to be used for each of the kernel
#         calculations

#     params_x : Dict[str, float]
#         a dictionary of parameters to be used for calculating the
#         kernel function for X

#     params_y : Dict[str, float]
#         a dictionary of parameters to be used for calculating the
#         kernel function for Y

#     Returns
#     -------
#     cka_value : float
#         the normalized hsic value.

#     Notes
#     -----

#         This is a metric that is similar to the correlation, [0,1]
#     """
#     # kernel matrix
#     Kx = covariance_matrix(kernel, params_x, X, X)
#     Ky = covariance_matrix(kernel, params_y, Y, Y)

#     Kx = centering(Kx)
#     Ky = centering(Ky)

#     hsic_value = np.sum(Kx * Ky)
#     if bias:
#         bias = 1 / (Kx.shape[0] ** 2)
#     else:
#         bias = 1 / (Kx.shape[0] - 1) ** 2
#     return bias * hsic_value


# def nhsic_cka(
#     X: np.ndarray,
#     Y: np.ndarray,
#     kernel: Callable,
#     params_x: Dict[str, float],
#     params_y: Dict[str, float],
# ) -> float:
#     """Normalized HSIC (Tangent Kernel Alignment)

#     A normalized variant of HSIC method which divides by
#     the HS-Norm of each dataset.

#     Parameters
#     ----------
#     X : jax.numpy.ndarray
#         the input value for one dataset

#     Y : jax.numpy.ndarray
#         the input value for the second dataset

#     kernel : Callable
#         the kernel function to be used for each of the kernel
#         calculations

#     params_x : Dict[str, float]
#         a dictionary of parameters to be used for calculating the
#         kernel function for X

#     params_y : Dict[str, float]
#         a dictionary of parameters to be used for calculating the
#         kernel function for Y

#     Returns
#     -------
#     cka_value : float
#         the normalized hsic value.

#     Notes
#     -----

#         This is a metric that is similar to the correlation, [0,1]

#     References
#     ----------
#     """
#     # calculate hsic normally (numerator)
#     # Pxy = hsic(X, Y, kernel, params_x, params_y)

#     # # calculate denominator (normalize)
#     # Px = np.sqrt(hsic(X, X, kernel, params_x, params_x))
#     # Py = np.sqrt(hsic(Y, Y, kernel, params_y, params_y))

#     # # print(Pxy, Px, Py)

#     # # kernel tangent alignment value (normalized hsic)
#     # cka_value = Pxy / (Px * Py)
#     Kx = covariance_matrix(kernel, params_x, X, X)
#     Ky = covariance_matrix(kernel, params_y, Y, Y)

#     Kx = centering(Kx)
#     Ky = centering(Ky)

#     cka_value = np.sum(Kx * Ky) / np.linalg.norm(Kx) / np.linalg.norm(Ky)

#     return cka_value


# def nhsic_nbs(
#     X: np.ndarray,
#     Y: np.ndarray,
#     kernel: Callable,
#     params_x: Dict[str, float],
#     params_y: Dict[str, float],
# ) -> float:
#     """Normalized Bures Similarity (NBS)

#     A normalized variant of HSIC method which divides by
#     the HS-Norm of the eigenvalues of each dataset.

#     ..math::
#         \\rho(K_x, K_y) = \\
#         \\text{Tr} ( K_x^{1/2} K_y K_x^{1/2)})^{1/2} \\
#         \ \\text{Tr} (K_x) \\text{Tr} (K_y)

#     Parameters
#     ----------
#     X : jax.numpy.ndarray
#         the input value for one dataset

#     Y : jax.numpy.ndarray
#         the input value for the second dataset

#     kernel : Callable
#         the kernel function to be used for each of the kernel
#         calculations

#     params_x : Dict[str, float]
#         a dictionary of parameters to be used for calculating the
#         kernel function for X

#     params_y : Dict[str, float]
#         a dictionary of parameters to be used for calculating the
#         kernel function for Y

#     Returns
#     -------
#     cka_value : float
#         the normalized hsic value.

#     Notes
#     -----

#         This is a metric that is similar to the correlation, [0,1]

#     References
#     ----------

#     @article{JMLR:v18:16-296,
#     author  = {Austin J. Brockmeier and Tingting Mu and Sophia Ananiadou and John Y. Goulermas},
#     title   = {Quantifying the Informativeness of Similarity Measurements},
#     journal = {Journal of Machine Learning Research},
#     year    = {2017},
#     volume  = {18},
#     number  = {76},
#     pages   = {1-61},
#     url     = {http://jmlr.org/papers/v18/16-296.html}
#     }
#     """
#     # calculate hsic normally (numerator)
#     # Pxy = hsic(X, Y, kernel, params_x, params_y)

#     # # calculate denominator (normalize)
#     # Px = np.sqrt(hsic(X, X, kernel, params_x, params_x))
#     # Py = np.sqrt(hsic(Y, Y, kernel, params_y, params_y))

#     # # print(Pxy, Px, Py)

#     # # kernel tangent alignment value (normalized hsic)
#     # cka_value = Pxy / (Px * Py)
#     Kx = covariance_matrix(kernel, params_x, X, X)
#     Ky = covariance_matrix(kernel, params_y, Y, Y)

#     Kx = centering(Kx)
#     Ky = centering(Ky)

#     # numerator
#     numerator = np.real(np.linalg.eigvals(np.dot(Kx, Ky)))

#     # clip rogue numbers
#     numerator = np.sqrt(np.clip(numerator, 0.0))

#     numerator = np.sum(numerator)

#     # denominator
#     denominator = np.sqrt(np.trace(Kx) * np.trace(Ky))

#     # return nbs value
#     return numerator / denominator


# def nhsic_ka(
#     X: np.ndarray,
#     Y: np.ndarray,
#     kernel: Callable,
#     params_x: Dict[str, float],
#     params_y: Dict[str, float],
# ) -> float:

#     Kx = covariance_matrix(kernel, params_x, X, X)
#     Ky = covariance_matrix(kernel, params_y, Y, Y)

#     cka_value = np.sum(Kx * Ky) / np.linalg.norm(Kx) / np.linalg.norm(Ky)

#     return cka_value


# def nhsic_cca(
#     X: np.ndarray,
#     Y: np.ndarray,
#     kernel: Callable,
#     params_x: Dict[str, float],
#     params_y: Dict[str, float],
#     epsilon: float = 1e-5,
#     bias: bool = False,
# ) -> float:
#     """Normalized HSIC (Tangent Kernel Alignment)

#     A normalized variant of HSIC method which divides by
#     the HS-Norm of each dataset.

#     Parameters
#     ----------
#     X : jax.numpy.ndarray
#         the input value for one dataset

#     Y : jax.numpy.ndarray
#         the input value for the second dataset

#     kernel : Callable
#         the kernel function to be used for each of the kernel
#         calculations

#     params_x : Dict[str, float]
#         a dictionary of parameters to be used for calculating the
#         kernel function for X

#     params_y : Dict[str, float]
#         a dictionary of parameters to be used for calculating the
#         kernel function for Y

#     Returns
#     -------
#     cka_value : float
#         the normalized hsic value.

#     Notes
#     -----

#         This is a metric that is similar to the correlation, [0,1]
#     """
#     n_samples = X.shape[0]

#     # kernel matrix
#     Kx = gram(kernel, params_x, X, X)
#     Ky = gram(kernel, params_y, Y, Y)

#     # center kernel matrices
#     Kx = centering(Kx)
#     Ky = centering(Ky)

#     K_id = np.eye(Kx.shape[0])
#     Kx_inv = np.linalg.inv(Kx + epsilon * n_samples * K_id)
#     Ky_inv = np.linalg.inv(Ky + epsilon * n_samples * K_id)

#     Rx = np.dot(Kx, Kx_inv)
#     Ry = np.dot(Ky, Ky_inv)

#     hsic_value = np.sum(Rx * Ry)

#     if bias:
#         bias = 1 / (Kx.shape[0] ** 2)
#     else:
#         bias = 1 / (Kx.shape[0] - 1) ** 2
#     return bias * hsic_value


# def _hsic_uncentered(
#     X: np.ndarray,
#     Y: np.ndarray,
#     kernel: Callable,
#     params_x: Dict[str, float],
#     params_y: Dict[str, float],
# ) -> float:
#     """A method to calculate the uncentered HSIC version"""
#     # kernel matrix
#     Kx = gram(kernel, params_x, X, X)
#     Ky = gram(kernel, params_y, Y, Y)

#     #
#     K = np.dot(Kx, Ky.T)

#     hsic_value = np.mean(K)

#     return hsic_value
