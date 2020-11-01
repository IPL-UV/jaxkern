import jax.numpy as np
import objax
from objax.typing import JaxArray


def zero_mean(x):
    """Mean Function"""
    return np.zeros(x.shape[0])


class ZeroMean(objax.Module):
    """Zero Mean Function"""

    def __init__(self, input_dim):

        self.zeros = np.zeros((input_dim,))

    def __call__(self, X: JaxArray) -> JaxArray:
        return self.zeros


class ConstantMean(objax.Module):
    """Constant Mean Function"""

    def __init__(self, input_dim, output_dim):
        self.c = objax.TrainVar(np.ones(input_dim))

    def __call__(self, X: JaxArray) -> JaxArray:
        return self.c.value * X


class LinearMean(objax.Module):
    """Linear Mean Function"""

    def __init__(self, input_dim, output_dim):
        self.w = objax.TrainVar(objax.random.normal((input_dim, output_dim)))
        self.b = objax.TrainVar(np.zeros(output_dim))

    def __call__(self, X: JaxArray) -> JaxArray:
        return np.dot(X, self.w.value) + self.b.value