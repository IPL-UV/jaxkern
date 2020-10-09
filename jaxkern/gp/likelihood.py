import jax
import objax
import jax.numpy as np


class GaussianLikelihood(objax.Module):
    def __init__(self):
        self.noise = objax.TrainVar(np.array([0.1]))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[-1], dtype=X.dtype)