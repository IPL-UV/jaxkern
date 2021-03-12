import jax.numpy as np
from chex import Array


def linear_kernel(x: Array, y: Array) -> Array:
    """Linear kernel function
    Takes two vectors and computes the dot product between them.

    Parameters
    ----------
    x : Array
        vector I, (n_features,)
    y : Array
        vector II, (n_features,)

    Returns
    -------
    k : Array
        the output, ()
    """
    return np.sum(x * y)
