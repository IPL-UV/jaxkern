from functools import partial
from typing import Callable, Dict, Tuple
import objax
import jax
import jax.numpy as np
from tensorflow_probability.substrates.jax import distributions as tfd
from jaxkern.gp.utils import get_factorizations


class BaseGP(objax.Module):
    def __init__(self, mean, kernel, noise: float = 0.1, jitter: float = 1e-5):

        # MEAN FUNCTION
        self.mean = mean

        # KERNEL Function
        self.kernel = kernel

        # noise level
        self.noise = objax.TrainVar(np.array([noise]))

        # jitter (make it correctly conditioned)
        self.jitter = jitter

    def forward(self, X: np.ndarray) -> np.ndarray:

        # mean function
        mu = self.mean(X)

        # kernel function
        cov = self.kernel(X, X)

        # noise model
        cov += jax.nn.softplus(self.noise.value) * np.eye(X.shape[0])

        # jitter
        cov += self.jitter * np.eye(X.shape[0])

        # calculate cholesky
        cov_chol = np.linalg.cholesky(cov)

        # gaussian process likelihood
        return tfd.MultivariateNormalTriL(loc=mu, scale_tril=cov_chol)

    def predict_f(self, X, return_std: bool = True):

        # projection kernel
        K_Xx = self.kernel(X, self.X_train_)

        # Calculate the Mean
        mu_y = np.dot(K_Xx, self.weights)

        if return_std:
            v = jax.scipy.linalg.cho_solve(self.L, K_Xx.T)

            K_xx = self.kernel(X, X)

            cov_y = K_xx - np.dot(K_Xx, v)

            return mu_y, cov_y

        else:
            return mu_y

    def predict_y(self, X, return_std: bool = True):

        # projection kernel
        K_Xx = self.kernel(X, self.X_train_)

        # Calculate the Mean
        mu_y = np.dot(K_Xx, self.weights)

        if return_std:
            v = jax.scipy.linalg.cho_solve(self.L, K_Xx.T)

            K_xx = self.kernel(X, X)

            cov_y = K_xx - np.dot(K_Xx, v)

            return mu_y, cov_y + jax.nn.softplus(self.noise.value) ** 2 * np.eye(
                cov_y.shape[0]
            )

        else:
            return mu_y
