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


def estimate_sigma_median_percent(X):
    dists = pdist_squareform(X, X)

    sigma = np.median(dists[np.nonzero(dists)])
    return sigma


def scotts_factor(X: np.ndarray) -> float:
    """Scotts Method to estimate the length scale of the 
    rbf kernel.

        factor = n**(-1./(d+4))

    Parameters
    ----------
    X : np.ndarry
        Input array

    Returns
    -------
    factor : float
        the length scale estimated

    """
    n_samples, n_features = X.shape
    return np.power(n_samples, -1 / (n_features + 4.0))


def silvermans_factor(X: np.ndarray) -> float:
    """Silvermans method used to estimate the length scale
    of the rbf kernel.

    factor = (n * (d + 2) / 4.)**(-1. / (d + 4)).

    Parameters
    ----------
    X : np.ndarray,
        Input array

    Returns
    -------
    factor : float
        the length scale estimated
    """
    n_samples, n_features = X.shape

    base = (n_samples * (n_features + 2.0)) / 4.0

    return np.power(base, -1 / (n_features + 4.0))


# def estimate_sigma_k_median(X, percent=0.3):
#     sigma = np.median(sqeuclidean_distance(X, X))
#     return sigma
