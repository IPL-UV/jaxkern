import functools

import jax
import jax.numpy as np

from jaxkern.dist import sqeuclidean_distance


@functools.partial(jax.jit, static_argnums=(0))
def gram(func, params, x, y):
    """Computes the gram matrix

    Parameters
    ----------
    func : Callable
        a callable function
    params : Dict
        the parameters needed for the kernel
    x : np.ndarray
        input dataset (n_samples, n_features)
    y : np.ndarray
        other input dataset (n_samples, n_features)
    
    Returns
    -------
    mat : np.ndarray
        the gram matrix.
    
    Examples
    --------

    >>> gram(kernel_rbf, {"gamma": 1.0}, X, Y)
    """
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(params, x1, y1))(y))(x)


@functools.partial(jax.jit, static_argnums=(0))
def covariance_matrix(kernel_func, params, x, y):
    mapx1 = jax.vmap(
        lambda x, y: kernel_func(params, x, y), in_axes=(0, None), out_axes=0
    )
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


@functools.partial(jax.jit, static_argnums=(0))
def linear_kernel(params, x, y):
    """Linear kernel
    
    .. math:: k_i = \sum_i^N x_i-y_i
    """
    return np.sum(x * y)


@jax.jit
def rbf_kernel(params, x, y):
    return np.exp(-params["gamma"] * sqeuclidean_distance(x, y))


# ARD Kernel
@jax.jit
def ard_kernel(params, x, y):

    # divide by the length scale
    x = x / params["length_scale"]
    y = y / params["length_scale"]

    # return the ard kernel
    return params["var_f"] * np.exp(-sqeuclidean_distance(x, y))


# Rational Quadratic Kernel
@jax.jit
def rq_kernel(params, x, y):

    # divide by the length scale
    x = x / params["length_scale"]
    y = y / params["length_scale"]

    # return the ard kernel
    return params["var_f"] * np.exp(1 + sqeuclidean_distance(x, y)) ** (
        -params["scale_mixture"]
    )
