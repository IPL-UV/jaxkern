from jaxkern.utils import centering
from typing import Callable
import math
import jax
import jax.numpy as np
import objax
from jaxkern.dependence import nhsic_cka
from jaxkern.dist import distmat, sqeuclidean_distance
from jaxkern.kernels.linear import Linear, linear_kernel
from jaxkern.kernels.approx import RBFSampler
from jaxkern.kernels.stationary import RBF
from jaxkern.kernels.utils import kernel_matrix


class HSIC(objax.Module):
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
            return hsic_v_statistic(K_x, K_y)
        else:
            return hsic_u_statistic(K_x, K_y)


class CKA(objax.Module):
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
            numerator = hsic_v_statistic(K_x, K_y)
            denominator = hsic_v_statistic(K_x, K_x) * hsic_v_statistic(K_y, K_y)
        else:
            numerator = hsic_u_statistic(K_x, K_y)
            denominator = hsic_u_statistic(K_x, K_x) * hsic_u_statistic(K_y, K_y)
        return numerator / np.sqrt(denominator)


class HSICRBF(objax.Module):
    def __init__(self, sigma_x: Callable, sigma_y: Callable, bias=False) -> None:
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.bias = bias

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute kernel matrices
        sigma_x = self.sigma_x(X, X)
        sigma_y = self.sigma_y(Y, Y)
        K_x = RBF(length_scale=sigma_x, variance=1.0)(X, X)
        K_y = RBF(length_scale=sigma_y, variance=1.0)(Y, Y)

        # center matrices
        K_x = centering(K_x)
        K_y = centering(K_y)

        # calculate hsic value
        hsic_value = np.sum(K_x * K_y)

        if self.bias == True:
            bias = 1 / (K_x.shape[0] ** 2)
        else:
            bias = 1 / (K_x.shape[0] - 1) ** 2
        return bias * hsic_value


class CKARBF(objax.Module):
    def __init__(self, sigma_x: Callable, sigma_y: Callable, bias=False) -> None:
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.bias = bias

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute kernel matrices
        sigma_x = self.sigma_x(X, X)
        sigma_y = self.sigma_y(Y, Y)
        K_x = RBF(length_scale=sigma_x, variance=1.0)(X, X)
        K_y = RBF(length_scale=sigma_y, variance=1.0)(Y, Y)

        # center matrices
        K_x = centering(K_x)
        K_y = centering(K_y)

        # calculate hsic value
        return np.sum(K_x * K_y) / np.linalg.norm(K_x) / np.linalg.norm(K_y)


class HSICRBFSampler(objax.Module):
    def __init__(
        self,
        n_rff: int = 100,
        length_scale_X: float = 2.0,
        length_scale_Y: float = 2.0,
        seed=(123, 42),
    ) -> None:
        self.n_rff = n_rff
        self.kernel_X = RBFSampler(
            n_rff=self.n_rff, length_scale=length_scale_X, center=True, seed=seed[0]
        )
        self.kernel_Y = RBFSampler(
            n_rff=self.n_rff, length_scale=length_scale_Y, center=True, seed=seed[1]
        )

    def __call__(self, X, Y):

        # calculate projection matrices
        Z_x = self.kernel_X(X)
        Z_y = self.kernel_Y(Y)

        # calculate centered hsic value
        return hsic_v_statistic_rff(Z_x, Z_y)


class CKARBFSampler(objax.Module):
    def __init__(
        self,
        n_rff: int = 100,
        length_scale_X: float = 2.0,
        length_scale_Y: float = 2.0,
        seed=(123, 42),
    ) -> None:
        self.n_rff = n_rff
        self.kernel_X = RBFSampler(
            n_rff=self.n_rff, length_scale=length_scale_X, center=True, seed=seed[0]
        )
        self.kernel_Y = RBFSampler(
            n_rff=self.n_rff, length_scale=length_scale_Y, center=True, seed=seed[1]
        )

    def __call__(self, X, Y):

        # calculate projection matrices
        Z_x = self.kernel_X(X)
        Z_y = self.kernel_Y(Y)

        # calculate centered hsic value
        numerator = hsic_v_statistic_rff(Z_x, Z_y)
        denominator = hsic_v_statistic_rff(Z_x, Z_x) * hsic_v_statistic_rff(Z_y, Z_y)
        return numerator / np.sqrt(denominator)


def hsic_u_statistic(K_x: np.ndarray, K_y: np.ndarray) -> np.ndarray:
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
    return a * A + b * B - c * C


def hsic_u_statistic_fast(K_x: np.ndarray, K_y: np.ndarray) -> np.ndarray:
    """
    Calculate the unbiased statistic (vectorized)
    """
    n_samples = K_x.shape[0]

    K_xd = K_x - np.diag(np.diag(K_x))
    K_yd = K_y - np.diag(np.diag(K_y))
    K_xy = K_xd @ K_yd

    # Term 1
    a = 1 / n_samples / (n_samples - 3)
    A = np.trace(K_xy)

    # Term 2
    b = a / (n_samples - 1) / (n_samples - 2)
    B = np.sum(K_xd) * np.sum(K_yd)

    # Term 3
    c = (a * 2) / (n_samples - 2)
    C = np.sum(K_xy)
    return a * A + b * B - c * C


def hsic_v_statistic(K_x, K_y):
    n_samples = K_x.shape[0]
    A = np.einsum("ij,ij->", K_x, K_y) / n_samples ** 2
    B = np.einsum("ij,kl->", K_x, K_y) / n_samples ** 4
    C = np.einsum("ij,ik->", K_x, K_y) / n_samples ** 3
    return A + B - 2 * C


def hsic_v_statistic_rff(Z_x, Z_y):
    n_samples = Z_x.shape[0]
    Z_x = Z_x - np.mean(Z_x, axis=0)
    Z_y = Z_y - np.mean(Z_y, axis=0)
    featCov = np.dot(Z_x.T, Z_y) / n_samples
    return np.linalg.norm(featCov) ** 2


def hsic_v_statistic_faster(K_x, K_y):
    n_samples = K_x.shape[0]
    K_xc = centering(K_x)
    K_yc = centering(K_y)
    return np.einsum("ij,ij->", K_xc, K_yc) / n_samples ** 2
