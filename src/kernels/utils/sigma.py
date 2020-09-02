import jax
import jax.numpy as np

from src.kernels.dist import pdist_squareform


def init_sigma_estimator(method="median", percent=None):

    if method == "median":

        return estimate_sigma_median

    return None


def estimate_sigma_median(X: np.ndarray, Y: np.ndarray) -> float:
    # compute distance matrix
    dists = pdist_squareform(X, Y)

    # remove non-zero elements
    dists = dists[np.nonzero(dists)]

    # get the median value
    sigma = np.median(dists)

    return sigma


def estimate_sigma_median_kth(X: np.ndarray, Y: np.ndarray, percent=0.3) -> float:

    # compute distance matrix
    dists = pdist_squareform(X, Y)

    # find the kth distance
    sigma = kth_distance(dists=dists, percent=percent)

    # median distances
    sigma = np.median(sigma)
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


def kth_distance(dists: np.ndarray, percent: float) -> np.ndarray:

    # kth distance calculation (50%)
    kth_sample = int(percent * dists.shape[0])

    # take the Kth neighbours of that distance
    k_dist = np.sort(dists)[:, kth_sample]

    return k_dist
