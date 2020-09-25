import jax
import jax.numpy as np

from jaxkern.dependence import nhsic_cka
from jaxkern.dist import distmat, sqeuclidean_distance
from jaxkern.kernels import linear_kernel


def rv_coeff(X: jax.numpy.ndarray, Y: jax.numpy.ndarray) -> float:
    """Calculates the RV coefficient

    This stands for the rho-Vector component and it is a non-linear
    extension to the Pearson correlation coefficient.

    .. math::
        :nowrap:

        \\begin{equation}
        \\rho V(\mathbf{x,y}) = \\
           \\frac{\\text{Tr}\left( \mathbf{xx}^\\top \\
           \mathbf{yy}^\\top \\right)}{\\
           \sqrt{\\text{Tr}\left( \\
           \mathbf{xx}^\\top \\right)^2\\
           \\text{Tr}\\left( \mathbf{yy}^\\top \\
           \\right)^2}}
        \\end{equation}

    where 
    :math:`\mathbf{x},\mathbf{y} \in \mathbb{R}^{N \\times D}`

    Parameters
    ----------
    X : jax.numpy.ndarray
        the input array, (n_samples, n_features)

    Y : jax.numpy.ndarray 
        the input array, (n_samples, m_features)

    Returns
    -------
    coeff : float
        the rv coefficient

    Notes
    -----

        This is simply the HSIC method but with a linear kernel.

    References
    ----------

    .. [1] Josse & Holmes, *Measuring Multivariate Association and Beyond*,
           Statistics Surveys, 2016, Volume 10, pg. 132-167

    """
    return nhsic_cka(X, Y, linear_kernel, {}, {})


def rv_coeff_features(X, Y):
    """Calculates the RV coefficient in the feature space

    This stands for the rho-Vector component and it is a non-linear
    extension to the Pearson correlation coefficient.

    .. math::
        :nowrap:

        \\begin{equation}
        \\rho V(\mathbf{x,y}) = \\
           \\frac{\\text{Tr}\left( \mathbf{x^\\top x} \\
           \mathbf{y^\\top y} \\right)}{\\
           \sqrt{\\text{Tr}\left( \\
           \mathbf{x^\\top x} \\right)^2\\
           \\text{Tr}\\left( \mathbf{y^\\top y} \\
           \\right)^2}}
        \\end{equation}

    where 
    :math:`\mathbf{x},\mathbf{y} \in \mathbb{R}^{N \\times D}`

    Parameters
    ----------
    X : jax.numpy.ndarray
        the input array, (n_samples, n_features)

    Y : jax.numpy.ndarray
        the input array, (n_samples, m_features)

    Returns
    -------
    coeff : float
        the rv coefficient

    Notes
    -----

        Sometimes this can be more efficient/effective if the
        number of features is greater than the number of samples.

    References
    ----------

    .. [1] Josse & Holmes, *Measuring Multivariate Association and Beyond*,
           Statistics Surveys, 2016, Volume 10, pg. 132-167

    """
    return nhsic_cka(X.T, Y.T, linear_kernel, {}, {})


def distance_corr(X: jax.numpy.ndarray, sigma=1.0) -> float:
    """Distance correlation"""
    X = distmat(sqeuclidean_distance, X, X)
    X = np.exp(-X / (2 * sigma ** 2))
    return np.mean(X)


def energy_distance(X: np.ndarray, Y: np.ndarray, sigma=1.0) -> float:
    """Distance correlation"""
    n_samples, m_samples = X.shape[0], Y.shape[0]
    a00 = -1.0 / (n_samples * n_samples)
    a11 = -1.0 / (m_samples * m_samples)
    a01 = 1.0 / (n_samples * m_samples)
    # X = distmat(sqeuclidean_distance, X, X)
    # X = np.exp(-X / (2 * sigma ** 2))

    # calculate distances
    dist_x = sqeuclidean_distance(X, X)
    dist_y = sqeuclidean_distance(Y, Y)
    dist_xy = sqeuclidean_distance(X, Y)

    return 2 * a01 * dist_xy + a00 * dist_x + a11 * dist_y