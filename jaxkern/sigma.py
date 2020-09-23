import jax
import jax.numpy as np

from jaxkern.dist import pdist_squareform


def estimate_sigma_median(X: np.ndarray, Y: np.ndarray) -> float:
    """Estimate sigma using the median distance

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


def estimate_sigma_mean_kth(
    X: np.ndarray, Y: np.ndarray, percent: float = 0.3
) -> float:
    """Estimates the sigma using the mean kth distance

    This calculates the sigma value using the kth percent
    of the distances. THe median value of that is the
    new sigma value.

    Parameters
    ----------
    dists : jax.numpy.ndarray
        the distance matrix already calculate (n_samples, n_samples)

    k : int
        the kth value from the (default=0.15)

    Returns
    -------
    kth_dist : jax.numpy.ndarray
        the neighbours up to the kth distance
    """

    # find the kth distance
    sigma = _estimate_sigma_kth(X=X, percent=percent)

    # median distances
    sigma = np.mean(sigma)
    return sigma


def estimate_sigma_median_kth(
    X: np.ndarray, Y: np.ndarray, percent: float = 0.3
) -> float:
    """Estimates the sigma using the median kth distance

    This calculates the sigma value using the kth percent
    of the distances. THe median value of that is the
    new sigma value.

    Parameters
    ----------
    dists : jax.numpy.ndarray
        the distance matrix already calculate (n_samples, n_samples)

    k : int
        the kth value from the (default=0.15)

    Returns
    -------
    kth_dist : jax.numpy.ndarray
        the neighbours up to the kth distance
    """

    # find the kth distance
    sigma = _estimate_sigma_kth(X=X, percent=percent)

    # median distances
    sigma = np.median(sigma)
    return sigma


def _estimate_sigma_kth(
    X: np.ndarray, Y: np.ndarray, percent: float = 0.3
) -> np.ndarray:
    """Private function to compute kth percent sigma."""
    # compute distance matrix
    dists = pdist_squareform(X, Y)

    # find the kth distance
    sigma = kth_percent_distance(dists=dists, percent=percent)
    return sigma


def scotts_factor(X: np.ndarray) -> float:
    """Scotts Method to estimate the length scale of the
    rbf kernel.

    .. math::

        \\sigma = n^{-\\frac{1}{d+4}}

    Parameters
    ----------
    X : jax.numpy.ndarry
        Input array

    Returns
    -------
    sigma : float
        the length scale estimated

    References
    ----------
    .. [1] Scott et al, *Multivariate Density Estimation:
        Theory, Practice, and Visualization*, New York, John Wiley, 1992
    """
    n_samples, n_features = X.shape
    return np.power(n_samples, -1 / (n_features + 4.0))


def silvermans_factor(X: np.ndarray) -> float:
    """Silvermans method used to estimate the length scale
    of the rbf kernel.

    .. math::

        \\sigma = \\frac{n(d + 2)}{4}^{-\\frac{1}{d + 4}}.

    Parameters
    ----------
    X : jax.numpy.ndarray,
        Input array (n_samples, n_features)

    Returns
    -------
    sigma : float
        the length scale estimated

    References
    ----------
    .. [1] Silverman, B. W., *Density Estimation for Statistics
         and Data Analysis*, London: Chapman and Hall., (1986)
    """
    n_samples, n_features = X.shape

    base = (n_samples * (n_features + 2.0)) / 4.0

    return np.power(base, -1 / (n_features + 4.0))


def kth_percent_distance(dists: np.ndarray, k: float = 0.3) -> np.ndarray:
    """kth percent distance in a gram matrix

    This calculates the kth percent in an (NxN) matrix.
    It sorts all distance values and then retrieves the
    kth value as a percentage of the number of samples.

    Parameters
    ----------
    dists : jax.numpy.ndarray
        the distance matrix already calculate (n_samples, n_samples)

    k : int
        the kth value from the (default=0.15)

    Returns
    -------
    kth_dist : jax.numpy.ndarray
        the neighbours up to the kth distance
    """
    # kth distance calculation (50%)
    kth_sample = int(k * dists.shape[0])

    # take the Kth neighbours of that distance
    k_dist = np.sort(dists)[:, kth_sample]

    return k_dist
