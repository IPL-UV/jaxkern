import jax.numpy as np
import objax
from objax.typing import JaxArray
from jaxkern.kernels.utils import kernel_matrix
from jaxkern.kernels.base import Kernel


class Linear(Kernel):
    """Linear Kernel

    Note
    ----
        They don't have to have the same number of samples but
        they do need to have the same number of features.
    """

    def __init__(self, input_dim: int = 1):

        self.input_dim = objax.TrainRef(np.array([input_dim]))

    def __call__(self, X: JaxArray, Y: JaxArray) -> JaxArray:
        """Kernel matrix for linear kernel

        Parameters
        ----------
        X : JaxArray
            dataset I, (n_samples, n_features)
        Y : JaxArray
            dataset II, (m_samples, n_features)

        Returns
        -------
        kernel_mat : JaxArray
            the kernel matrix, (n_samples, m_samples
        """
        return kernel_matrix(
            linear_kernel,
            X,
            Y,
        )


def linear_kernel(x: JaxArray, y: JaxArray) -> JaxArray:
    """Linear kernel function
    Takes two vectors and computes the dot product between them.

    Parameters
    ----------
    x : JaxArray
        vector I, (n_features,)
    y : JaxArray
        vector II, (n_features,)

    Returns
    -------
    k : JaxArray
        the output, ()
    """
    return np.sum(x * y)
