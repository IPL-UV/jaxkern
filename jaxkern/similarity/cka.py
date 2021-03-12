from chex import dataclass, Array
from jaxkern.kernels.utils import covariance_matrix
from jaxkern.similarity.hsic import hsic_v_statistic_trace
from typing import Callable
import jax.numpy as jnp


def cka_biased(
    kernel_f: Callable, params_X: dataclass, params_Y: dataclass, X: Array, Y: Array
):
    """Centered Kernel Alignment
    Parameters
    ----------
    kernel_f : Callable,
    params_X : dataclass
    params_Y: dataclass
    X : Array
    Y : Array

    Returns
    -------
    cka_score : Array
    """

    # calculate the kernel matrices
    K_x = covariance_matrix(params=params_X, func=kernel_f, x=X, y=X)
    K_y = covariance_matrix(params=params_Y, func=kernel_f, x=Y, y=Y)

    # calculate hsic
    numerator = hsic_v_statistic_trace(K_x, K_y)
    denominator = hsic_v_statistic_trace(K_x, K_x) * hsic_v_statistic_trace(K_y, K_y)

    # calculate cka
    return numerator / jnp.sqrt(denominator)
