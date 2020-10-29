import jax.numpy as np
import objax
import jax
from jaxkern.dist import sqeuclidean_distance, distmat
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
