from jaxkern.utils import centering
from typing import Callable
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

        # center matrices
        K_x = centering(K_x)
        K_y = centering(K_y)

        # calculate centered hsic value
        return np.sum(K_x * K_y) / np.linalg.norm(K_x) / np.linalg.norm(K_y)


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
        length_scale: float = 2.0,
        center: bool = True,
        bias: bool = True,
    ) -> None:
        self.n_rff = n_rff
        self.bias = bias
        self.kernel_X = RBFSampler(
            n_rff=self.n_rff, length_scale=length_scale, center=True
        )
        self.kernel_Y = RBFSampler(
            n_rff=self.n_rff, length_scale=length_scale, center=True
        )

    def __call__(self, X, Y):

        # calculate projection matrices
        Zx = self.kernel_X(X)
        Zy = self.kernel_Y(Y)

        # calculate kernel matrices
        Rxy = Zx.T @ Zy

        # calculate hsic value
        hsic_value = np.sum(Rxy * Rxy.T)

        if self.bias == True:
            bias = 1 / (X.shape[0] ** 2)
        else:
            bias = 1 / (X.shape[0] - 1) ** 2
        return bias * hsic_value


class CKARBFSampler(objax.Module):
    def __init__(
        self, n_rff: int = 100, length_scale: float = 2.0, center: bool = True
    ) -> None:
        self.n_rff = n_rff
        self.kernel_X = RBFSampler(
            n_rff=self.n_rff, length_scale=length_scale, center=True
        )
        self.kernel_Y = RBFSampler(
            n_rff=self.n_rff, length_scale=length_scale, center=True
        )

    def __call__(self, X, Y):

        # calculate projection matrices
        Zx = self.kernel_X(X)
        Zy = self.kernel_Y(Y)

        # calculate kernel matrices
        Rxy = Zx.T @ Zy

        # calculate hsic value
        return np.sum(Rxy * Rxy.T) / np.linalg.norm(Rxy) / np.linalg.norm(Rxy.T)