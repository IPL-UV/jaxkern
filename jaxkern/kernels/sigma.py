import jax
import jax.numpy as np
from objax.typing import JaxArray

from jaxkern.dist import pdist_squareform
from jaxkern.utils import ensure_min_eps


def init_ard_params(X, Y, method: str = "median"):

    return None


def estimate_sigma_median(X: JaxArray, Y: JaxArray) -> JaxArray:
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
    # dists = dists[np.nonzero(dists)]

    # get the median value
    dists = np.clip(dists, a_min=1e-5, a_max=np.inf)
    median_dist = np.median(dists)
    sigma = median_dist / np.sqrt(2.0)

    return sigma


def estimate_sigma_mean_kth(X: JaxArray, Y: JaxArray, percent: float = 0.3) -> JaxArray:
    """Estimates the sigma using the mean kth distance

    This calculates the sigma value using the kth percent
    of the distances. THe median value of that is the
    new sigma value.

    Parameters
    ----------
    X : JaxArray
        dataset I (n_samples, n_features)
    Y : JaxArray
        dataset II (n_samples, n_features)
    k : int
        the kth value from the (default=0.15)

    Returns
    -------
    kth_dist : jax.numpy.ndarray
        the neighbours up to the kth distance
    """

    # find the kth distance
    dists = _estimate_sigma_kth(X=X, Y=Y, percent=percent)

    # median distances
    sigma = np.mean(dists[np.nonzero(dists)])

    return sigma


def estimate_sigma_median_kth(
    X: JaxArray, Y: JaxArray, percent: float = 0.3
) -> JaxArray:
    """Estimates the sigma using the median kth distance

    This calculates the sigma value using the kth percent
    of the distances. THe median value of that is the
    new sigma value.

    Parameters
    ----------
    X : JaxArray
        dataset I (n_samples, n_features)
    Y : JaxArray
        dataset II (n_samples, n_features)
    k : int
        the kth value from the (default=0.3)

    Returns
    -------
    kth_dist : jax.numpy.ndarray
        the neighbours up to the kth distance
    """

    # find the kth distance
    dists = _estimate_sigma_kth(X=X, Y=Y, percent=percent)

    # median distances
    sigma = np.median(dists[np.nonzero(dists)])
    return sigma


def _estimate_sigma_kth(
    X: np.ndarray, Y: np.ndarray, percent: float = 0.3
) -> np.ndarray:
    """Private function to compute kth percent sigma."""
    # compute distance matrix
    dists = pdist_squareform(X, Y)

    # find the kth distance
    sigma = kth_percent_distance(dists=dists, k=percent)
    return sigma


def scotts_factor(X: JaxArray) -> JaxArray:
    """Scotts Method to estimate the length scale of the
    rbf kernel.

    .. math::

        \\sigma = n^{-\\frac{1}{d+4}}

    Parameters
    ----------
    X : JaxArray
        Input array

    Returns
    -------
    sigma : JaxArray
        the length scale estimated

    References
    ----------
    .. [1] Scott et al, *Multivariate Density Estimation:
        Theory, Practice, and Visualization*, New York, John Wiley, 1992
    """
    n_samples, n_features = X.shape
    return np.power(n_samples, -1 / (n_features + 4.0))


def silvermans_factor(X: JaxArray) -> JaxArray:
    """Silvermans method used to estimate the length scale
    of the rbf kernel.

    .. math::

        \\sigma = \\frac{n(d + 2)}{4}^{-\\frac{1}{d + 4}}.

    Parameters
    ----------
    X : JaxArray
        Input array (n_samples, n_features)

    Returns
    -------
    sigma : JaxArray
        the length scale estimated

    References
    ----------
    .. [1] Silverman, B. W., *Density Estimation for Statistics
         and Data Analysis*, London: Chapman and Hall., (1986)
    """
    n_samples, n_features = X.shape

    base = (n_samples * (n_features + 2.0)) / 4.0

    return np.power(base, -1 / (n_features + 4.0))


def kth_percent_distance(dists: JaxArray, k: float = 0.3) -> JaxArray:
    """kth percent distance in a gram matrix

    This calculates the kth percent in an (NxN) matrix.
    It sorts all distance values and then retrieves the
    kth value as a percentage of the number of samples.

    Parameters
    ----------
    dists : JaxArray
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


def gamma_to_sigma(gamma: float = JaxArray) -> JaxArray:
    """Convert sigma to gamma

    .. math::

        \\sigma = \\frac{1}{\\sqrt{2 \\gamma}}
    """
    return ensure_min_eps(np.sqrt(1.0 / (2 * gamma)))


def sigma_to_gamma(sigma: float = JaxArray) -> JaxArray:
    """Convert sigma to gamma

    .. math::

        \\gamma = \\frac{1}{2 \\sigma^2}
    """
    return ensure_min_eps(1.0 / (2 * sigma ** 2))
