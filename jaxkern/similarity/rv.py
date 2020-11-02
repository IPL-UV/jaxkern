from typing import Callable

import jax
import jax.numpy as np
import objax
from objax.typing import JaxArray
from jaxkern.kernels.linear import Linear, linear_kernel
from jaxkern.kernels.stationary import RBF
from jaxkern.similarity.hsic import CKA, hsic_u_statistic, hsic_v_statistic


class RVCoeff(CKA):
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
    kernel_X : Callable
        the kernel function used to 

    Y : jax.numpy.ndarray 
        the input array, (n_samples, m_features)

    Returns
    -------
    coeff : float
        the rv coefficient

    Notes
    -----

        This traditional approach is the RVCoefis simply the HSIC method but with a linear kernel.

    References
    ----------

    .. [1] Josse & Holmes, *Measuring Multivariate Association and Beyond*,
           Statistics Surveys, 2016, Volume 10, pg. 132-167

    """

    def __init__(
        self,
        kernel_X: Callable[[JaxArray, JaxArray], JaxArray] = Linear(),
        kernel_Y: Callable[[JaxArray, JaxArray], JaxArray] = Linear(),
        bias: bool = True,
    ):
        super().__init__(kernel_X=kernel_X, kernel_Y=kernel_Y, bias=bias)


def rv_coeff(X: JaxArray, Y: JaxArray) -> JaxArray:
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


def rv_coeff_feat(X: JaxArray, Y: JaxArray, bias: bool = True) -> JaxArray:
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

    # compute kernel matrices
    K_x = linear_kernel(X, X)
    K_y = linear_kernel(Y, Y)

    # calculate centered hsic value
    if bias is True:
        numerator = hsic_u_statistic(K_x, K_y)
        denominator = hsic_v_statistic(K_x, K_x) * hsic_v_statistic(K_y, K_y)
    else:
        numerator = hsic_u_statistic(K_x, K_y)
        denominator = hsic_u_statistic(K_x, K_x) * hsic_u_statistic(K_y, K_y)
    return numerator / np.sqrt(denominator)
