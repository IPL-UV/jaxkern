from typing import Tuple

import jax
import jax.numpy as np
import objax

from chex import Array, dataclass
from jaxkern.kernels.base import Kernel

# from tensorflow_probability.substrates.jax import distributions as tfd


@dataclass
class SpectralRBFParams:
    length_scale: Array
    n_ff: int


def spectral_rbf_projection(
    key, params: SpectralRBFParams, X: Array, n_rff: int
) -> Array:

    # generate rff weights
    W, b = generate_rff_weights(key, X.shape[1], n_rff, params.length_scale)

    # calculate projection matrix
    Z = np.cos(np.dot(X, W) + b[None, :])

    # normalize projection matrix
    Z *= np.sqrt(2.0) / np.sqrt(n_rff)

    # center samples-wise
    Z = Z - np.mean(Z, axis=0)

    # return projection matrix
    return Z


def spectral_rbf_kernel(key, params, X, n_rff):

    # calculate projection matrix
    Z = spectral_rbf_projection(key, params, X, n_rff)

    # calculate kernel matrix
    return np.dot(Z, Z.T)


def generate_rff_weights(
    key,
    n_features: int,
    n_rff: int,
    length_scale: float = 1.0,
) -> Tuple[Array, Array]:
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
    W : Array
        the weights for the kernel matrix, (n_features, n_rff)
    b : Array
        the bias for the kernel matrix, (n_rff,)
    """

    # weights
    W = jax.random.normal(key=key, shape=(n_features, n_rff))

    W *= 1.0 / length_scale

    # bias
    b = jax.random.uniform(key=key, shape=(n_rff,))
    b *= 2 * np.pi

    return W, b
