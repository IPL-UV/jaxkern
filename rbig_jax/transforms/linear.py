import collections
from functools import partial

import jax
import jax.numpy as np

RotParams = collections.namedtuple("Params", ["projection"])


@jax.jit
def compute_projection(X: np.ndarray) -> np.ndarray:
    """Compute PCA projection matrix
    Using SVD, this computes the PCA components for
    a dataset X and computes the projection matrix
    needed to do the PCA decomposition.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        the data to calculate to PCA projection matrix
    
    Returns
    -------
    VT : np.ndarray, (n_features, n_features)
        the projection matrix (V.T) for the PCA decomposition

    Notes
    -----
    Can find the original implementation here:
    https://bit.ly/2EBDV9o
    """

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
