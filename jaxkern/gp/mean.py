import objax
import jax.numpy as np


def zero_mean(x):
    """Mean Function"""
    return np.zeros(x.shape[0])


class ZeroMean(objax.Module):
    def __init__(self):
        pass

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[-1], dtype=X.dtype)


class LinearMean(objax.Module):
    def __init__(self, input_dim, output_dim):
        self.w = objax.TrainVar(objax.random.normal((input_dim, output_dim)))
        self.b = objax.TrainVar(np.zeros(output_dim))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X.T, self.w.value) + self.b.value