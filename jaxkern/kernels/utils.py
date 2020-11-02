from typing import Callable, Dict

import jax
import jax.numpy as np
from objax.typing import JaxArray


def kernel_matrix(
    kernel_func: Callable[[JaxArray, JaxArray], JaxArray],
    X: JaxArray,
    Y: JaxArray,
) -> JaxArray:
    """Computes the kernel matrix given a kernel function

    Given a `Callable` function, we can use the `jax.vmap` function
    to calculate the kernel matrix as the function applied to each
    of the points. K_ij = k(x_i, y_j)

    Parameters
    ----------
    kernel_func : Callable[[JaxArray, JaxArray], JaxArray]
        a callable function (kernel or distance)
    X : JaxArray
        dataset I (n_samples, n_features)
    Y : JaxArray
        dataset II (m_samples, n_features)

    Returns
    -------
    K : JaxArray
        the kernel matrix, (n_samples, m_samples)

    Notes
    -----

        This is also known as the gram matrix.

    Examples
    --------

    >>> kernel_matrix(kernel_rbf, X, Y)
    """
    mv = jax.vmap(lambda x1, y1: kernel_func(x1, y1), in_axes=(0, None), out_axes=0)
    mm = jax.vmap(lambda x2, y2: mv(x2, y2), in_axes=(None, 0), out_axes=1)
    return mm(X, Y)


def centering(kernel_mat: JaxArray) -> JaxArray:
    """Calculates the centering matrix for the kernel
    Particularly useful in unsupervised kernel methods like
    HSIC and MMD.

    Parameters
    ----------
    kernel_mat : JaxArray
        PSD kernel matrix, (n_samples, n_samples)

    Returns
    -------
    centered_kernel_mat : JaxArray
        centered PSD kernel matrix, (n_samples, n_samples)
    """
    # get centering matrix
    H = centering_matrix(np.shape(kernel_mat)[0])

    # center kernel matrix
    kernel_mat = np.einsum("ij,jk,kl->il", H, kernel_mat, H)

    return kernel_mat


def centering_matrix(n_samples: int) -> JaxArray:
    """Calculates the centering matrix H

    Parameters
    ----------
    n_samples : int

    Returns
    -------
    H : JaxArray
        the centering matrix, (n_samples, n_samples)
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
