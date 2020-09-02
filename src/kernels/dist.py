import functools
from typing import Callable

import jax
import jax.numpy as np


@jax.jit
def euclidean_distance(x: np.array, y: np.array) -> float:
    return np.sqrt(np.sum((x - y) ** 2))


# Squared Euclidean Distance Formula
@jax.jit
def sqeuclidean_distance(x: np.array, y: np.array) -> float:
    return np.sum((x - y) ** 2)


# Manhattan Distance
@jax.jit
def manhattan_distance(x: np.array, y: np.array) -> float:
    return np.sum(np.abs(x - y))


# distance matrix
@functools.partial(jax.jit, static_argnums=(0))
def distmat(func: Callable, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(y))(x)


# pdist squareform
def pdist_squareform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return distmat(sqeuclidean_distance, x, y)

