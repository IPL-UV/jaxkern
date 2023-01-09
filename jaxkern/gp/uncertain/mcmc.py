from typing import Tuple

import jax
import jax.numpy as np
import numpyro.distributions as dist
import objax
from objax.typing import JaxArray

from jaxkern.gp.predictive import predictive_mean


class MCTransform(objax.Module):
    def __init__(self, model, jitted: bool = False, seed: int = 123):
        self.n_features = model.X_train_.shape[-1]

        self.key = jax.random.PRNGKey(seed)
        f = jax.vmap(jax.partial(predictive_mean, model))

        transform = jax.vmap(jax.partial(_transform, f))

        if jitted:
            transform = jax.jit(transform)

        self.transform = transform

    def forward(self, X, Xcov, n_samples: int) -> Tuple[JaxArray, JaxArray]:

        self.wm, self.wc = get_mc_weights(n_samples)

        key, self.key = jax.random.split(self.key, 2)
        # generate mc samples
        mc_samples = get_mc_sigma_points(
            key, self.n_features, n_samples=(X.shape[0], n_samples)
        )
        # form sigma points from unit sigma-points
        mean_f, var_f = self.transform(X, Xcov, mc_samples)

        return mean_f, var_f


def _transform(mean_f, X, Xcov, mc_samples):

    # define the constants
    wm = 1.0 / mc_samples.shape[0]
    wc = 1.0 / (mc_samples.shape[0] - 1.0)

    # form sigma points from unit sigma-points
    x_ = X[:, None] + np.linalg.cholesky(Xcov) @ mc_samples.T

    fx_ = mean_f(x_.T)

    # output mean
    mean_f = (wm * fx_).sum(axis=0)

    # output covariance
    dfx_ = (fx_ - mean_f)[:, None]

    cov_f = wc * np.dot(dfx_.T, dfx_)

    return mean_f, np.diag(cov_f)


def get_mc_weights(n_samples: int = 100) -> Tuple[float, float]:
    """Generate normalizers for MCMC samples"""
    mean = 1.0 / n_samples
    cov = 1.0 / (n_samples - 1)
    return mean, cov


def get_mc_sigma_points(
    key, n_features: int, n_samples: Tuple[int] = (100,)
) -> JaxArray:
    """Generate MCMC samples from a normal distribution"""
    sigma_dist = dist.Normal(loc=np.zeros(shape=(n_features,)), scale=n_features)

    return sigma_dist.sample(key, sample_shape=n_samples)