from typing import Callable

import jax
import jax.numpy as np
from chex import Array
from jaxkern.kernels.linear import Linear, linear_kernel
from jaxkern.kernels.stationary import RBF
from jaxkern.similarity.hsic import (
    CKA,
    hsic_u_statistic_einsum,
    hsic_v_statistic_einsum,
)


def pearson_corr_coeff(x, y):
    """Pearson Correlation Coefficient

    .. math::

        \\rho = \\frac{\\sum (x - m_x) (y - m_y)}
                 {\\sqrt{\\sum (x - m_x)^2 \\sum (y - m_y)^2}}

    where :math:`m_x` is the mean of the vector :math:`x` and :math:`m_y` is
    the mean of the vector :math:`y`.

    Parameters
    ----------
    x : Array,
        vector I of  inputs, (n_samples,)
    y : Array,
        vector II of inputs, (n_samples,)

    Returns
    ------
    rho : Array
        pearson correlation coefficient, ()
    """

    # remove the mean
    x -= np.mean(x)
    y -= np.mean(y)

    # calculate the covariance
    cov_xy = np.dot(x, y)

    # calculate the roots
    den = np.sqrt(np.sum(x ** 2) * np.sum(y ** 2))

    return cov_xy / den


def sub_pearson_corr_coeff(X, Y):
    """Pearson Correlation Coefficient applied Dimension-wise

    .. math::

        \\rho = \\frac{1}{D} \\sum_{d}^D\\rho\\
            (\\mathbf{X}_d,\\mathbf{Y}_d)

    where :math:`\\mathbf{X,Y} \in \mathbb{R}^{N \\times D}`

    Parameters
    ----------
    x : Array,
        vector I of  inputs, (n_samples, n_features)
    y : Array,
        vector II of inputs, (n_samples, n_features)

    Returns
    ------
    rho : Array
        pearson correlation coefficient, ()
    """
    n_features = X.shape[1]

    rhos = jax.vmap(pearson_corr_coeff, in_axes=(0, 1))(X.T, Y.T)

    return np.sum(rhos ** 2) / n_features


def rv_coeff(X: Array, Y: Array, center: bool = False, bias: bool = True) -> Array:
    """Calculates the RV coefficient

    This stands for the rho-Vector component and it is a multivariate
    extension to the Pearson correlation coefficient.

    .. math::
        :nowrap:

        \\begin{equation}
        \\rho V(\\mathbf{X,Y}) = \\
           \\frac{\\text{Tr}\\left( \\mathbf{XX}^\\top \\
           \\mathbf{YY}^\\top \\right)}{\\
           \\sqrt{\\text{Tr}\left( \\
           \\mathbf{XX}^\\top \\right)^2\\
           \\text{Tr}\\left( \\mathbf{YY}^\\top \\
           \\right)^2}}
        \\end{equation}

    where 
    :math:`\mathbf{X},\mathbf{Y} \in \mathbb{R}^{N \\times D}`

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
    if center:
        X -= np.mean(X, axis=1)
        Y -= np.mean(Y, axis=1)

    # compute kernel matrices
    K_x = X @ X.T
    K_y = Y @ Y.T

    return np.einsum("ij,ij->", K_x, K_y) / np.linalg.norm(K_x) / np.linalg.norm(K_y)


def rv_coeff_feat(X: Array, Y: Array, center: bool = False, bias: bool = True) -> Array:
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
    bias : bool
        whether to do the biased or unbiased statistic

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
    if center:
        X = X - np.mean(X, axis=0)
        Y = Y - np.mean(Y, axis=0)
    # compute kernel matrices
    Sigma_XY = X.T @ Y
    Sigma_XX = X.T @ X
    Sigma_YY = Y.T @ Y

    return (
        np.linalg.norm(Sigma_XY) ** 2
        / np.linalg.norm(Sigma_XX)
        / np.linalg.norm(Sigma_YY)
    )
