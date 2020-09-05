import jax
import jax.numpy as np

from jaxkern.kernels import linear_kernel
from jaxkern.dependence import nhsic_cka
from jaxkern.dist import distmat, sqeuclidean_distance


def rv_coeff(X, Y):

    return nhsic_cka(X, Y, linear_kernel, {}, {})


def rv_coeff_features(X, Y):

    return nhsic_cka(X.T, Y.T, linear_kernel, {}, {})


def distance_corr(X, sigma=1.0) -> float:
    X = distmat(sqeuclidean_distance, X, X)
    X = np.exp(-X / (2 * sigma ** 2))
    return np.mean(X)
