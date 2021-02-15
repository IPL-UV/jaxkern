import functools
from typing import Callable

import jax
import jax.numpy as np
from objax.typing import JaxArray


def sqeuclidean_distance(x: JaxArray, y: JaxArray) -> JaxArray:
    """Squared Euclidean Distance
    Performs the squared euclidean distance metric between two vectors.

    Parameters
    ----------
    x : JaxArray
        x vector (n_features,)
    y : JaxArray
        y vector (n_features,)

    Returns
    -------
    dist : JaxArray
        the distance between x and y, (,)
    """
    return np.sum((x - y) ** 2)


def euclidean_distance(x: JaxArray, y: JaxArray) -> JaxArray:
    """Euclidean Distance
    Performs the Euclidean distance metric between two vectors.

    Parameters
    ----------
    x : JaxArray
        x vector (n_features,)
    y : JaxArray
        y vector (n_features,)

    Returns
    -------
    dist : JaxArray
        the distance between x and y, (,)
    """
    return np.sqrt(sqeuclidean_distance(x, y))


# @jax.jit
def manhattan_distance(x: JaxArray, y: JaxArray) -> JaxArray:
    """Manhattan Distance
    Performs the Manhattan distance metric between two vectors.

    Parameters
    ----------
    x : JaxArray
        x vector (n_features,)
    y : JaxArray
        y vector (n_features,)

    Returns
    -------
    dist : JaxArray
        the distance between x and y, (,)
    """
    return np.sum(np.abs(x - y))


def distmat(func: Callable, X: JaxArray, Y: JaxArray) -> JaxArray:
    """Distance Matrix

    Parameters
    ----------
    func : Callable[[JaxArray, JaxArray], JaxArray]
        a callable function that takes two vectors and returns a scalar
    X : JaxArray
        a stack of vectors (n_samples, n_features)
    Y : JaxArray
        a stack of vectors (m_samples, n_features)

    Returns
    -------
    dist_mat : JaxArray
        a distance matrix, (n_samples, m_samples)
    """
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(Y))(X)


def pdist_squareform(X: JaxArray, Y: JaxArray) -> JaxArray:
    """Squared Euclidean Distance Matrix
    Computes the squared euclidean distance matrix given two
    batches of input vectors.

    Parameters
    ----------
    X : JaxArray
        a stack of vectors (n_samples, n_features)
    Y : JaxArray
        a stack of vectors (m_samples, n_features)

    Returns
    -------
    dist_mat : JaxArray
        a distance matrix, (n_samples, m_samples)

    Notes
    -----
    This is equivalent to the scipy commands

    >>> from scipy.spatial.distance import pdist, squareform
    >>> dists = squareform(pdist(X, metric='sqeuclidean')
    """
    return distmat(sqeuclidean_distance, X, Y)
