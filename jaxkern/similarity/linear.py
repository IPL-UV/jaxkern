import jax
import jax.numpy as np

from src.kernels import covariance_matrix, linear_kernel
from src.kernels.dependence import centered_kernel_alignment


def rv_coeff(X, Y):

    return centered_kernel_alignment(X, Y, linear_kernel, {}, {})


def rv_coeff_features(X, Y):

    return centered_kernel_alignment(X.T, Y.T, linear_kernel, {}, {})
