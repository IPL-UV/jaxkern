from typing import Callable

import jax
import jax.numpy as np


def gaussian_kernel(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)


def kde_pdf(
    x: np.ndarray, samples: np.ndarray, bandwidth: float, kernel: Callable
) -> np.ndarray:
    n_samples = samples.shape[0]
    dists = (x - samples) / bandwidth
    density = kernel(dists)
    return density.sum() / bandwidth / n_samples


def kde_pdf_gaussian(
    x: np.ndarray, samples: np.ndarray, bandwidth: float
) -> np.ndarray:
    n_samples = samples.shape[0]
    dists = (x - samples) / bandwidth
    density = gaussian_kernel(dists)
    return density.sum() / bandwidth / n_samples


def kde_cdf_gaussian(
    x: np.ndarray, samples: np.ndarray, bandwidth: float
) -> np.ndarray:
    n_samples = samples.shape[0]

    # normalize samples
    low = (-np.inf - samples) / bandwidth
    x = (x - samples) / bandwidth

    # evaluate integral
    integral = jax.scipy.special.ndtr(x) - jax.scipy.special.ndtr(low)

    # normalize distribution
    x_cdf = integral.sum() / n_samples

    return x_cdf
