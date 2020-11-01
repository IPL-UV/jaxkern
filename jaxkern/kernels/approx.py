from typing import Tuple

import jax.numpy as np
import objax

from jaxkern.kernels.base import Kernel


class RBFSampler(objax.Module):
    """
    Random Fourier Features (RFF) for RBF Kernel
    """

    def __init__(
        self,
        n_rff: int = 100,
        length_scale: float = 2.0,
        center: bool = False,
        seed=123,
    ) -> None:
        self.n_rff = n_rff
        self.length_scale = objax.TrainVar(np.array([length_scale]))
        self.center = center
        self.rng = objax.RandomState(seed)

    def __call__(self, X: np.ndarray) -> np.ndarray:

        # sample weights
        W, b = self.get_weights(X.shape[1])

        # calculate projection matrix
        Z = np.cos(np.dot(X, W) + b)

        # normalize projection matrix
        Z = np.sqrt(2.0) / np.sqrt(self.n_rff) * Z

        if self.center == True:
            Z = Z - np.mean(Z, axis=0)

        # return projection matrix
        return Z

    def get_weights(self, n_features: int) -> Tuple[np.ndarray, np.ndarray]:

        # weights
        W = objax.random.normal(
            mean=0, stddev=1.0, shape=(n_features, self.n_rff), generator=self.rng
        ) * (1.0 / self.length_scale.value)

        b = 2 * np.pi * objax.random.uniform(shape=(1, self.n_rff), generator=self.rng)

        return W, b

    def calculate_kernel(self, X):
        """
        Explicitly Calculates the kernel matrix.
        """
        Z = self.__call__(X)

        return np.dot(Z, Z.T)
