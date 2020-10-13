from jaxkern.dist import sqeuclidean_distance
from typing import Callable, Dict
import jax
import jax.numpy as np
import objax

from jaxkern.kernels.utils import gram, covariance_matrix
from jaxkern.kernels.stationary import RBF
from jaxkern.utils import centering

jax_np = jax.numpy.ndarray


class MMD_PXPY(objax.Module):
    """
    Maximum Mean Discrepency (MMD) - PxPy
    """

    def __init__(
        self,
        kernel_X: Callable,
        kernel_Y: Callable,
        center: bool = False,
    ) -> None:
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.center = center

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute kernel matrices
        K_x = self.kernel_X(X, X)
        K_y = self.kernel_Y(Y, Y)

        # center matrices
        if self.center == True:
            K_x = centering(K_x)
            K_y = centering(K_y)

        # calculate hsic value
        A = np.mean(K_x * K_y)
        B = np.mean(np.mean(K_x, axis=0) * np.mean(K_y, axis=0))
        C = np.mean(K_x) * np.mean(K_y)

        return A - 2 * B + C


class MMD(objax.Module):
    """
    Maximum Mean Discrepency (MMD)

    Some advantages of this method is that you don't need to
    have the same number of samples as the PxPy method.
    """

    def __init__(
        self,
        kernel_X: Callable,
        kernel_Y: Callable,
        kernel_XY: Callable,
        center: bool = False,
        bias: bool = True,
    ) -> None:
        self.kernel_X = kernel_X
        self.kernel_Y = kernel_Y
        self.kernel_XY = kernel_XY
        self.center = center
        self.bias = bias

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute kernel matrices
        K_x = self.kernel_X(X, X)
        K_y = self.kernel_Y(Y, Y)
        K_xy = self.kernel_XY(X, Y)

        # center matrices
        if self.center == True:
            K_x = centering(K_x)
            K_y = centering(K_y)
            K_xy = centering(K_xy)

        # calculate hsic value
        if self.bias is True:
            mmd = mmd_v_statistic(K_x, K_y, K_xy)

        else:
            mmd = mmd_u_statistic(K_x, K_y, K_xy)

        # numerical error
        # mmd = np.clip(mmd, a_min=0.0, a_max=np.inf)

        return np.sqrt(mmd)


class MMD_PXPY_RBF(objax.Module):
    """
    Maximum Mean Discrepency (MMD) w. RBF Kernel - PxPy
    """

    def __init__(
        self, sigma_x: Callable, sigma_y: Callable, center: bool = False, bias=False
    ) -> None:
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.center = center

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # compute kernel matrices
        sigma_x = self.sigma_x(X, X)
        sigma_y = self.sigma_y(Y, Y)
        K_x = RBF(length_scale=sigma_x, variance=1.0)(X, X)
        K_y = RBF(length_scale=sigma_y, variance=1.0)(Y, Y)

        # center matrices
        if self.center == True:
            K_x = centering(K_x)
            K_y = centering(K_y)

        # calculate hsic value
        A = np.mean(K_x * K_y)
        B = np.mean(np.mean(K_x, axis=0) * np.mean(K_y, axis=0))
        C = np.mean(K_x) * np.mean(K_y)

        return A - 2 * B + C


class MMD_RBF(objax.Module):
    """
    Maximum Mean Discrepency (MMD)

    Some advantages of this method is that you don't need to
    have the same number of samples as the PxPy method.

    A disadvantage is that now you have to compute 3 kernels
    which means 3 kernel parameters and also 1 kernel extra
    over the PxPy method.
    """

    def __init__(
        self,
        sigma_X: Callable,
        sigma_Y: Callable,
        sigma_XY: Callable,
        center: bool = False,
        bias: bool = True,
    ) -> None:
        self.sigma_X = sigma_X
        self.sigma_Y = sigma_Y
        self.sigma_XY = sigma_XY
        self.center = center
        self.bias = bias

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        # get samples for constants
        n_samples, m_samples = X.shape[0], Y.shape[0]

        a00 = 1.0 / (n_samples * (n_samples - 1.0))
        a11 = 1.0 / (m_samples * (m_samples - 1.0))
        a01 = -1.0 / (n_samples * m_samples)

        # estimate sigma values
        σ_x = self.sigma_X(X, X)
        σ_y = self.sigma_Y(Y, Y)
        σ_xy = self.sigma_XY(X, Y)

        # kernel matrices
        K_x = RBF(variance=1.0, length_scale=σ_x)(X, X)
        K_y = RBF(variance=1.0, length_scale=σ_y)(Y, Y)
        K_xy = RBF(variance=1.0, length_scale=σ_xy)(X, Y)

        # center matrices
        if self.center == True:
            K_x = centering(K_x)
            K_y = centering(K_y)
            K_xy = centering(K_xy)

        # calculate hsic value
        if self.bias == True:
            mmd = np.mean(K_x) + np.mean(K_y) - 2 * np.mean(K_xy)

        else:
            mmd = (
                2 * a01 * np.mean(K_xy)
                + a00 * (np.sum(K_x) - n_samples)
                + a11 * (np.sum(K_y) - m_samples)
            )

        # numerical error
        mmd = np.clip(mmd, a_min=0.0, a_max=np.inf)

        return np.sqrt(mmd)


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
