import collections

from functools import partial
import jax
import jax.numpy as np

RotParams = collections.namedtuple("Params", ["projection"])


@jax.jit
def compute_projection(X: np.ndarray) -> np.ndarray:
    """Compute PCA projection"""

    # center the data
    X = X - np.mean(X, axis=0)

    # Compute SVD
    _, _, VT = np.linalg.svd(X, full_matrices=False, compute_uv=True)

    return VT.T


def init_pca_params(X):

    # compute projection matrix
    R = compute_projection(X)

    return (
        np.dot(X, R),
        RotParams(R),
        partial(forward_transform, R=R),
        partial(inverse_transform, R=R),
    )


@jax.jit
def forward_transform(X, R):
    return np.dot(X, R), np.zeros(X.shape)


@jax.jit
def inverse_transform(X, R):
    return np.dot(X, R.T)
