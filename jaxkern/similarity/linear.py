import jax
import jax.numpy as np

from jaxkern.kernels import covariance_matrix, linear_kernel
from jaxkern.kernels.dependence import nhsic_cka


def rv_coeff(X, Y):

    return nhsic_cka(X, Y, linear_kernel, {}, {})


def rv_coeff_features(X, Y):

    return nhsic_cka(X.T, Y.T, linear_kernel, {}, {})
