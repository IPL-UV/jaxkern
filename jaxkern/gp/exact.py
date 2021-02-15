from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
import objax
from objax.typing import JaxArray
from tensorflow_probability.substrates.jax import distributions as tfd

from jaxkern.gp.utils import get_factorizations

DEFAULT_VARIANCE_LOWER_BOUND = 1e-6


class ExactGP(objax.Module):
    """Exact GP

    Parameters
    ----------
    mean : Callable
        the gp mean function
    kernel : Callable
        the gp kernel function
    noise : float
        the noise likelihood
    jitter : float
        the 'jitter' to allow for better conditioning of the cholesky
    """

    def __init__(self, mean, kernel, noise: float = 0.1, jitter: float = 1e-5):

        # MEAN FUNCTION
        self.mean = mean

        # KERNEL Function
        self.kernel = kernel

        # noise level
        self.noise = objax.TrainVar(np.array([noise]))

        # jitter (make it correctly conditioned)
        self.jitter = jitter

    def forward(self, X: JaxArray) -> JaxArray:
        """Fits a GP distribution

        Parameters
        ----------
        X : JaxArray
            the input dataset

        Returns
        -------
        gp_dist :
            Returns a gp distribution.
        """
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

    def cache_factorizations(self, X: JaxArray, y: JaxArray) -> None:
        """Cache Factorizations
        The allows one to speed up the predictions

        Parameters
        ----------
        X : JaxArray
            input trained dataset, (n_samples, n_features)
        y : JaxArray
            output trained dataset, (n_samples,)
        """
        # calculate factorizations
        self.X_train_, self.y_train_ = X, y
        self.L, self.weights = get_factorizations(
            X, y, jax.nn.softplus(self.noise.value), self.mean, self.kernel
        )

    def predict_f(self, X, return_std: bool = True):
        """Predictive mean and variance on new data

        Parameters
        ----------
        X : JaxArray
            the dataset to do predictions, (n_samples, n_features)

        return_std : bool
            option to return the standard deviation or the covariance (default=True)

        Returns
        -------
        mu : JaxArray
            the predictive mean, (n_samples)
        var : JaxArray
            the predictive variance, (n_samples)
        cov : JaxArray
            the predictive covariance, (n_samples, n_samples)
        """
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
