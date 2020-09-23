import jax
import jax.numpy as np

from jaxkern.dist import pdist_squareform


def estimate_sigma_median(X: jax.numpy.ndarray, Y: jax.numpy.ndarray) -> float:
    """Estimate sigma using the median

    Parameters
    ----------
    X : jax.numpy.ndarray
        input data (n_samples, n_features)
    Y : jax.numpy.ndarray
        input data (n_samples, n_features)

    Returns
    -------
    sigma : float
        the estimated sigma
    """
    # compute distance matrix
    dists = pdist_squareform(X, Y)

    # remove non-zero elements
    dists = dists[np.nonzero(dists)]

    # get the median value
    sigma = np.median(dists)

    return sigma


def estimate_sigma_median_kth(
    X: jax.numpy.ndarray, Y: jax.numpy.ndarray, percent: float = 0.3
) -> float:

    # compute distance matrix
    dists = pdist_squareform(X, Y)

    # find the kth distance
    sigma = kth_distance(dists=dists, percent=percent)

    # median distances
    sigma = np.median(sigma)
    return sigma


def scotts_factor(X: jax.numpy.ndarray) -> float:
    """Scotts Method to estimate the length scale of the
    rbf kernel.

        factor = n**(-1./(d+4))

    Parameters
    ----------
    X : jax.numpy.ndarry
        Input array

    Returns
    -------
    factor : float
        the length scale estimated

    """
    n_samples, n_features = X.shape
    return np.power(n_samples, -1 / (n_features + 4.0))


def silvermans_factor(X: jax.numpy.ndarray) -> float:
    """Silvermans method used to estimate the length scale
    of the rbf kernel.

    factor = (n * (d + 2) / 4.)**(-1. / (d + 4)).

    Parameters
    ----------
    X : jax.numpy.ndarray,
        Input array

    Returns
    -------
    factor : float
        the length scale estimated
    """
    n_samples, n_features = X.shape

    base = (n_samples * (n_features + 2.0)) / 4.0

    return np.power(base, -1 / (n_features + 4.0))


def kth_distance(dists: jax.numpy.ndarray, percent: float) -> jax.numpy.ndarray:

    # kth distance calculation (50%)
    kth_sample = int(percent * dists.shape[0])

    # take the Kth neighbours of that distance
    k_dist = np.sort(dists)[:, kth_sample]

    return k_dist
