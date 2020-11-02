from typing import Tuple

import jax.numpy as np
import objax
from objax.typing import JaxArray

from jaxkern.kernels.base import Kernel


class RBFSampler(objax.Module):
    """Random Fourier Features (RFF) for RBF Kernel

    Parameters
    ----------
    n_rff : int
        number of random fourier features to approximate the
        kernel matrix, (default=100)
    length_scale : int
        the length scale for the RBF kernel, this is a trainable
        parameter (default=2.0)
    center : bool
        option to center the projection matrix sample-wise, (default=True)
    seed : int
        the random state for generating the features, (default=123)
    """

    def __init__(
        self,
        n_rff: int = 100,
        length_scale: float = 2.0,
        center: bool = False,
        seed: int = 123,
    ) -> None:
        self.n_rff = n_rff
        self.length_scale = objax.TrainVar(np.array([length_scale]))
        self.center = center
        self.rng = objax.RandomState(seed)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Calculates projection matrix

        Parameters
        ----------
        X : JaxArray
            the stack of features to calculate the projection
            (n_samples, n_features)

        Returns
        -------
        Z : JaxArray
            the projection matrix, (n_samples, n_rff)
        """

        # sample weights
        W, b = generate_rff_weights(
            X.shape[1], self.n_rff, self.length_scale.value, self.rng
        )

        # calculate projection matrix
        Z = np.cos(np.dot(X, W) + b[None, :])

        # normalize projection matrix
        Z *= np.sqrt(2.0) / np.sqrt(self.n_rff)

        # center samples-wise
        if self.center == True:
            Z = Z - np.mean(Z, axis=0)

        # return projection matrix
        return Z

    def calculate_kernel(self, X: JaxArray) -> JaxArray:
        """Calculates approximate RBF kernel matrix.

        Parameters
        ----------
        X : JaxArray
            the data to be used to calculate the kernel matrix, (n_samples, n_features)

        Returns
        -------
        kernel_mat : JaxArray
            the approximate kernel matrix, (n_samples, n_samples)
        """
        Z = self.__call__(X)

        return np.dot(Z, Z.T)


def generate_rff_weights(
    n_features: int, n_rff: int, length_scale: float = 1.0, seed: int = 123
) -> Tuple[JaxArray, JaxArray]:
    """Generate weights and bias
    Given a number of features and fourier features, it will
    generate a bias and

    Parameters
    ----------
    n_features : int
        number of features for the dataset
    n_rff : int
        the number of random fourier features
    seed : int
        the seed for the random generation

    Returns
    -------
    W : JaxArray
        the weights for the kernel matrix, (n_features, n_rff)
    b : JaxArray
        the bias for the kernel matrix, (n_rff,)
    """

    # weights
    W = objax.random.normal(
        mean=0, stddev=1.0, shape=(n_features, n_rff), generator=seed
    ) * (1.0 / length_scale)

    # bias
    b = 2 * np.pi * objax.random.uniform(shape=(n_rff,), generator=seed)

    return W, b
