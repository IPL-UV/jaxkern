import math
from typing import Callable, Optional

import jax
import jax.numpy as np
import objax
from objax.typing import JaxArray

from jaxkern.dist import distmat, sqeuclidean_distance
from jaxkern.kernels.approx import RBFSampler
from jaxkern.kernels.linear import linear_kernel
from jaxkern.kernels.stationary import rbf_kernel
from jaxkern.kernels.utils import kernel_matrix
from jaxkern.utils import centering


class HSIC(objax.Module):
    """Hilbert-Schmidt Independence Criterion

    Parameters
    ----------
    kernel_X : Callable
        the kernel matrix to be used for X
    kernel_Y : Callable
        the kernel matrix to be used for Y
    bias : bool
        the bias term for the hsic method; similar to covariance normalization
        (default = False)
    """

    def __init__(
        self, kernel_X: Callable, kernel_Y: Callable, bias: bool = True
    ) -> None:
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.bias = bias

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute kernel matrices
        K_x = self.kernel_X(X, X)
        K_y = self.kernel_Y(Y, Y)

        if self.bias is True:
            return hsic_v_statistic_einsum(K_x, K_y)
        else:
            return hsic_u_statistic_einsum(K_x, K_y)


class CKA(objax.Module):
    """Normalized Hilbert-Schmidt Independence Criterion

    Parameters
    ----------
    kernel_X : Callable
        the kernel matrix to be used for X
    kernel_Y : Callable
        the kernel matrix to be used for Y
    bias : bool
        the bias term for the hsic method; similar to covariance normalization
        (default = False)
    """

    def __init__(
        self, kernel_X: Callable, kernel_Y: Callable, bias: bool = True
    ) -> None:
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.bias = bias

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute kernel matrices
        K_x = self.kernel_X(X, X)
        K_y = self.kernel_Y(Y, Y)

        # calculate centered hsic value
        if self.bias is True:
            numerator = hsic_v_statistic_einsum(K_x, K_y)
            denominator = hsic_v_statistic_einsum(K_x, K_x) * hsic_v_statistic_einsum(
                K_y, K_y
            )
        else:
            numerator = hsic_u_statistic_einsum(K_x, K_y)
            denominator = hsic_u_statistic_einsum(K_x, K_x) * hsic_u_statistic_einsum(
                K_y, K_y
            )
        return numerator / np.sqrt(denominator)


class HSICRBF(objax.Module):
    def __init__(self, sigma_x: Callable, sigma_y: Callable, bias=False) -> None:
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.bias = bias
        self.kernel = jax.vmap(rbf_kernel)

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute kernel matrices
        sigma_x = self.sigma_x(X, X)
        sigma_y = self.sigma_y(Y, Y)
        kernelx_f = jax.partial(rbf_kernel, sigma_x, 1.0)
        kernely_f = jax.partial(rbf_kernel, sigma_y, 1.0)
        K_x = kernel_matrix(kernelx_f, X, X)
        K_y = kernel_matrix(kernely_f, Y, Y)

        if self.bias is True:
            return hsic_v_statistic_einsum(K_x, K_y)
        else:
            return hsic_u_statistic_einsum(K_x, K_y)

        # # center matrices
        # K_x = centering(K_x)
        # K_y = centering(K_y)

        # # calculate hsic value
        # hsic_value = np.sum(K_x * K_y)

        # if self.bias == True:
        #     bias = 1 / (K_x.shape[0] ** 2)
        # else:
        #     bias = 1 / (K_x.shape[0] - 1) ** 2
        # return bias * hsic_value


class CKARBF(objax.Module):
    def __init__(
        self,
        sigma_x: Callable,
        sigma_y: Callable,
        n_subsamples: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.bias = bias
        self.n_subsamples = n_subsamples

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute sigma values
        # compute kernel matrices
        self.sigma_x_ = self.sigma_x(X, X)
        self.sigma_y_ = self.sigma_y(Y, Y)

        # compute kernel matrices
        kernelx_f = jax.partial(rbf_kernel, self.sigma_x_, 1.0)
        kernely_f = jax.partial(rbf_kernel, self.sigma_y_, 1.0)
        K_x = kernel_matrix(kernelx_f, X, X)
        K_y = kernel_matrix(kernely_f, Y, Y)

        # calculate normalized hsic value
        if self.bias is True:
            numerator = hsic_v_statistic_einsum(K_x, K_y)
            denominator = hsic_v_statistic_einsum(K_x, K_x) * hsic_v_statistic_einsum(
                K_y, K_y
            )
        else:
            numerator = hsic_u_statistic_einsum(K_x, K_y)
            denominator = hsic_u_statistic_einsum(K_x, K_x) * hsic_u_statistic_einsum(
                K_y, K_y
            )

        return numerator / np.sqrt(denominator)


class HSICRBFSampler(objax.Module):
    def __init__(
        self,
        n_rff: int = 100,
        length_scale_X: float = 2.0,
        length_scale_Y: float = 2.0,
        n_subsamples: int = 1_000,
        seed=(123, 42),
    ) -> None:
        self.seed = seed
        self.n_rff = n_rff
        self.n_subsamples = n_subsamples
        self.length_scale_X = length_scale_X
        self.length_scale_Y = length_scale_Y

    def __call__(self, X, Y):

        length_scale_X = self.length_scale_X(
            X[: self.n_subsamples], X[: self.n_subsamples]
        )
        length_scale_Y = self.length_scale_Y(
            Y[: self.n_subsamples], Y[: self.n_subsamples]
        )

        self.kernel_X = RBFSampler(
            n_rff=self.n_rff,
            length_scale=length_scale_X,
            center=False,
            seed=self.seed[0],
        )
        self.kernel_Y = RBFSampler(
            n_rff=self.n_rff,
            length_scale=length_scale_Y,
            center=False,
            seed=self.seed[1],
        )

        # calculate projection matrices
        Z_x = self.kernel_X(X)
        Z_y = self.kernel_Y(Y)

        # calculate centered hsic value
        return hsic_v_statistic_rff(Z_x, Z_y)


class CKARBFSampler(HSICRBFSampler):
    def __call__(self, X, Y):

        length_scale_X = self.length_scale_X(
            X[: self.n_subsamples], X[: self.n_subsamples]
        )
        length_scale_Y = self.length_scale_Y(
            Y[: self.n_subsamples], Y[: self.n_subsamples]
        )

        self.kernel_X = RBFSampler(
            n_rff=self.n_rff,
            length_scale=length_scale_X,
            center=False,
            seed=self.seed[0],
        )
        self.kernel_Y = RBFSampler(
            n_rff=self.n_rff,
            length_scale=length_scale_Y,
            center=False,
            seed=self.seed[1],
        )

        # calculate projection matrices
        Z_x = self.kernel_X(X)
        Z_y = self.kernel_Y(Y)

        # calculate centered hsic value
        numerator = hsic_v_statistic_rff(Z_x, Z_y)
        denominator = hsic_v_statistic_rff(Z_x, Z_x) * hsic_v_statistic_rff(Z_y, Z_y)

        return numerator / np.sqrt(denominator)


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


def hsic_u_statistic_dot(K_x: JaxArray, K_y: JaxArray) -> JaxArray:
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
    n_samples = K_x.shape[0]
    K_xc = centering(K_x)
    K_yc = centering(K_y)
    return np.einsum("ij,ij->", K_xc, K_yc) / n_samples ** 2


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
