import jax
import jax.numpy as np

from src.kernels.dist import pdist_squareform


def init_sigma_estimator(method="median", percent=None):

    if method == "median":

        return estimate_sigma_median

    return None


def estimate_sigma_median(X):
    dists = pdist_squareform(X, X)
    sigma = np.median(dists[np.nonzero(dists)])
    return sigma


# def estimate_sigma_k_median(X, percent=0.3):
#     sigma = np.median(sqeuclidean_distance(X, X))
#     return sigma
