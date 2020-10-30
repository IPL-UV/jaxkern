import functools
from typing import Callable

import jax
import jax.numpy as np
from objax.typing import JaxArray


# @jax.jit
def sqeuclidean_distance(x: JaxArray, y: JaxArray) -> JaxArray:
    return np.sum((x - y) ** 2)


# @jax.jit
def euclidean_distance(x: JaxArray, y: np.array) -> JaxArray:
    return np.sqrt(sqeuclidean_distance(x, y))


# @jax.jit
def manhattan_distance(x: JaxArray, y: JaxArray) -> JaxArray:
    return np.sum(np.abs(x - y))


# @functools.partial(jax.jit, static_argnums=(0))
def distmat(func: Callable, x: JaxArray, y: JaxArray) -> JaxArray:
    """distance matrix"""
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)


# pdist squareform
# @jax.jit
def pdist_squareform(x: JaxArray, y: JaxArray) -> JaxArray:
    """squared euclidean distance matrix

    Notes
    -----
    This is equivalent to the scipy commands

    >>> from scipy.spatial.distance import pdist, squareform
    >>> dists = squareform(pdist(X, metric='sqeuclidean')
    """
    return distmat(sqeuclidean_distance, x, y)
