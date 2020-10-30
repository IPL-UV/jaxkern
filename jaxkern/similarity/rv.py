from typing import Callable

import jax
import jax.numpy as np
import objax

from jaxkern.dependence import nhsic_cka
from jaxkern.dist import distmat, sqeuclidean_distance
from jaxkern.kernels.linear import Linear, linear_kernel
from jaxkern.kernels.stationary import RBF
from jaxkern.kernels.utils import kernel_matrix
from jaxkern.utils import centering


class RVCoeff(CKA):
    def __init__(self):
        self.kernel_X = Linear()
        self.kernel_Y = Linear()


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
