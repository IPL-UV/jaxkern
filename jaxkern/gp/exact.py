from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
import objax
from tensorflow_probability.substrates.jax import distributions as tfd

from jaxkern.gp.utils import get_factorizations

DEFAULT_VARIANCE_LOWER_BOUND = 1e-6


class ExactGP(objax.Module):
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
        cov += (
            np.clip(
                jax.nn.softplus(self.noise.value) ** 2,
                DEFAULT_VARIANCE_LOWER_BOUND,
                10.0,
            )
            * np.eye(X.shape[0])
        )

        # jitter
        cov += self.jitter * np.eye(X.shape[0])

        # calculate cholesky
        cov_chol = np.linalg.cholesky(cov)

        # gaussian process likelihood
        return tfd.MultivariateNormalTriL(loc=mu.squeeze(), scale_tril=cov_chol)

    def cache_factorizations(self, X: np.ndarray, y: np.ndarray) -> None:

        # calculate factorizations
        self.X_train_, self.y_train_ = X, y
        self.L, self.weights = get_factorizations(
            X, y, jax.nn.softplus(self.noise.value), self.mean, self.kernel
        )

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
