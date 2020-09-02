import jax.numpy as np
from src.kernels.dist import distmat, sqeuclidean_distance


def distance_corr(X, sigma=1.0) -> float:
    X = distmat(sqeuclidean_distance, X, X)
    X = np.exp(-X / (2 * sigma ** 2))
    return np.mean(X)
