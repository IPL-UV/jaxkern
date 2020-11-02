import jax
import jax.numpy as np
import objax
from objax.typing import JaxArray

from jaxkern.dist import distmat, sqeuclidean_distance
from jaxkern.kernels.base import Kernel


class Stationary(Kernel):
    """
    Stationary Kernel
    """

    def __init__(
        self, variance: float = 1.0, length_scale: float = 1.0, input_dim: int = 1
    ):
        if isinstance(variance, int) or isinstance(variance, float):
            self.variance = objax.TrainVar(np.array([variance]))
        else:
            self.variance = objax.TrainVar(variance)

        if isinstance(length_scale, int) or isinstance(length_scale, float):
            self.length_scale = objax.TrainVar(np.array([length_scale]))
        else:
            self.length_scale = objax.TrainVar(length_scale)

        self.input_dim = input_dim

    def squared_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        return distmat(
            sqeuclidean_distance,
            self.scale(X),
            self.scale(Y),
        )

    def scale(self, X: np.ndarray) -> np.ndarray:
        return X / np.clip(
            jax.nn.softplus(self.length_scale.value), a_min=0.0, a_max=10.0
        )

    def Kdiag(self, X: np.ndarray) -> np.ndarray:
        return np.abs(self.variance.value) * np.ones(X.shape[0])


class RBF(Stationary):
    """
    Radial Basis Function (RBF) or Squared Exponential / Gaussian Kernel
    """

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        variance = jax.nn.softplus(self.variance.value)

        # numerical stability
        dists = np.clip(self.squared_distance(X, Y), a_min=0.0, a_max=np.inf)

        return variance * np.exp(-dists / 2.0)


class RationalQuadratic(Stationary):
    """
    Rational Quadratic Kernel
    """

    def __init__(
        self,
        variance: float = 1.0,
        length_scale: float = 1.0,
        alpha: float = 1.0,
        input_dim: int = 1,
    ):
        super().__init__(
            variance=variance, length_scale=length_scale, input_dim=input_dim
        )
        self.alpha = objax.TrainVar(np.array([alpha]))

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:

        variance = jax.nn.softplus(self.variance.value)

        # numerical stability
        dists = np.clip(self.squared_distance(X, Y), a_min=0.0, a_max=np.inf)

        alpha = variance = jax.nn.softplus(self.alpha.value)

        return variance * (1 + 0.5 * dists / alpha) ** (-alpha)


def rbf_kernel(
    length_scale: float, variance: float, x: JaxArray, y: JaxArray
) -> JaxArray:
    """Automatic Relevance Determination (ARD) Kernel.

    This is an RBF kernel with a variable length scale. It
    *should* be the most popular kernel of all of the kernel 
    methods.

    .. math::

        k(\mathbf{x,y}) = \\
           \\exp \left( -\\frac{1}{2} \\
           \left|\left|\\frac{\mathbf{x}}{\sigma}\\
            - \\frac{\mathbf{y}}{\sigma} \\right|\\
                \\right|^2_2 \\right) 


    Parameters
    ----------
    length_scale : float
        the length scale for the scaling
    variance : float
        the function variance scaling
    x : jax.numpy.ndarray
        input dataset (n_samples, n_features)
    y : jax.numpy.ndarray
        other input dataset (n_samples, n_features)

    Returns
    -------
    kernel_mat : jax.numpy.ndarray
        the kernel matrix (n_samples, n_samples)
        
    References
    ----------
    .. [1] David Duvenaud, *Kernel Cookbook*
    """
    # divide by the length scale
    x = x / length_scale
    y = y / length_scale

    # return the ard kernel
    return variance * np.exp(-0.5 * sqeuclidean_distance(x, y))
