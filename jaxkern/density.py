from typing import Callable
import objax
from objax.typing import JaxArray
import jax
import jax.numpy as np


class KDE(objax.Module):
    """Kernel Density Estimation (KDE) class

    Parameters
    ----------
    samples : JaxArray
        representative samples to calculate the KDE
    kernel : Callable[[JaxArray], JaxArray]
        a callable kernel function class
    n_grid_points : int
        the number of grid points for the representative points for
        the PDF and CDF
    """

    def __init__(
        self,
        samples: JaxArray,
        kernel_f: Callable[[JaxArray], JaxArray],
        n_grid_points: int = 50,
    ):
        self.samples = samples
        self.kernel_f = kernel_f
        self.n_grid_points = n_grid_points

    def pdf(self, X: JaxArray) -> JaxArray:
        raise NotImplementedError()

    def cdf(self, X: JaxArray) -> JaxArray:
        raise NotImplementedError()

    def icdf(self, X: JaxArray) -> JaxArray:
        raise NotImplementedError()


class KDEGauss(KDE):
    """Gaussian Kernel Density Estimation (KDE) class

    Parameters
    ----------
    bandwidth : Callable[[JaxArray], JaxArray]
        a callable kernel function class
    """

    def __init__(self, samples: JaxArray, bandwidth: float):
        self.samples = samples
        self.bandwidth = bandwidth

    def pdf(self, X: JaxArray) -> JaxArray:
        return kde_pdf_gaussian(X, self.samples, self.bandwidth)

    def cdf(self, X: JaxArray) -> JaxArray:
        return kde_pdf_gaussian(X, self.samples, self.bandwidth)

    def icdf(self, X: JaxArray) -> JaxArray:
        raise NotImplementedError()


def gaussian_kernel(x: np.ndarray) -> np.ndarray:
    """Gaussian kernel function for KDE"""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def kde_pdf(
    x: np.ndarray, samples: np.ndarray, bandwidth: float, kernel: Callable
) -> np.ndarray:
    """PDF Estimation for KDE"""
    n_samples = samples.shape[0]
    dists = (x - samples) / bandwidth
    density = kernel(dists)
    return density.sum() / bandwidth / n_samples


def kde_pdf_gaussian(
    x: np.ndarray, samples: np.ndarray, bandwidth: float
) -> np.ndarray:
    """PDF Estimation for Gaussian KDE"""
    n_samples = samples.shape[0]
    dists = (x - samples) / bandwidth
    density = gaussian_kernel(dists)
    return density.sum() / bandwidth / n_samples


def kde_cdf_gaussian(
    x: np.ndarray, samples: np.ndarray, bandwidth: float
) -> np.ndarray:
    """Exact CDF Estimation for Gaussian KDE"""
    n_samples = samples.shape[0]

    # normalize samples
    low = (-np.inf - samples) / bandwidth
    x = (x - samples) / bandwidth

    # evaluate integral
    integral = jax.scipy.special.ndtr(x) - jax.scipy.special.ndtr(low)

    # normalize distribution
    x_cdf = integral.sum() / n_samples

    return x_cdf
