from typing import Callable, Dict
import jax
import jax.numpy as np


def kernel_matrix(
    kernel_func: Callable,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Computes the covariance matrix.

    Given a function `Callable` and some `params`, we can
    use the `jax.vmap` function to calculate the gram matrix
    as the function applied to each of the points.

    Parameters
    ----------
    kernel_func : Callable
        a callable function (kernel or distance)
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Returns
    -------
    mat : jax.ndarray
        the gram matrix.

    Notes
    -----

        There is little difference between this function
        and `gram`

    See Also
    --------
    jax.kernels.gram

    Examples
    --------

    >>> kernel_matrix(kernel_rbf, X, Y)
    """
    mapx1 = jax.vmap(lambda x, y: kernel_func(x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)


def centering(kernel_mat: jax.numpy.ndarray) -> jax.numpy.ndarray:
    """Calculates the centering matrix for the kernel"""
    n_samples = np.shape(kernel_mat)[0]

    identity = np.eye(n_samples)

    H = identity - (1.0 / n_samples) * np.ones((n_samples, n_samples))

    kernel_mat = np.einsum("ij,jk,kl->il", H, kernel_mat, H)

    return kernel_mat


def centering_matrix(n_samples: int) -> np.ndarray:
    """
    Calculates the centering matrix
    """
    return np.eye(n_samples) - (1.0 / n_samples) * np.ones((n_samples, n_samples))


def gram(
    func: Callable,
    params: Dict,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Computes the gram matrix.

    Given a function `Callable` and some `params`, we can
    use the `jax.vmap` function to calculate the gram matrix
    as the function applied to each of the points.

    Parameters
    ----------
    func : Callable
        a callable function (kernel or distance)
    params : Dict
        the parameters needed for the kernel
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
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


def covariance_matrix(
    func: Callable,
    params: Dict[str, float],
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Computes the covariance matrix.

    Given a function `Callable` and some `params`, we can
    use the `jax.vmap` function to calculate the gram matrix
    as the function applied to each of the points.

    Parameters
    ----------
    kernel_func : Callable
        a callable function (kernel or distance)
    params : Dict
        the parameters needed for the kernel
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Returns
    -------
    mat : jax.ndarray
        the gram matrix.

    Notes
    -----

        There is little difference between this function
        and `gram`

    See Also
    --------
    jax.kernels.gram

    Examples
    --------

    >>> covariance_matrix(kernel_rbf, {"gamma": 1.0}, X, Y)
    """
    mapx1 = jax.vmap(lambda x, y: func(params, x, y), in_axes=(0, None), out_axes=0)
    mapx2 = jax.vmap(lambda x, y: mapx1(x, y), in_axes=(None, 0), out_axes=1)
    return mapx2(x, y)
