from jaxkern.gp.uncertain.moment import MomentTransform
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax as tfp

dist = tfp.distributions
import objax
from chex import Array, dataclass
from objax.typing import JaxArray

from jaxkern.gp.predictive import predictive_mean


@dataclass
class MCMomentTransform(MomentTransform):
    n_features: int
    n_samples: int
    seed: int

    def __post_init__(
        self,
    ):
        self.rng_key = jr.PRNGKey(self.seed)
        self.z = dist.Normal(
            loc=jnp.zeros((self.n_features,)), scale=jnp.ones(self.n_features)
        )
        wm, wc = get_mc_weights(self.n_samples)
        self.wm, self.wc = wm, wc

    def predict_f(self, f, x, x_cov, rng_key):

        # sigma points
        sigma_pts = self.z.sample((self.n_samples,), rng_key)

        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_mc_samples = x[:, None] + L @ sigma_pts.T
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_mc = jax.vmap(f, in_axes=1, out_axes=1)(x_mc_samples)
        # print(x_mc_samples.shape, y_mu_mc.shape)

        # mean of mc samples
        # (P,) = (P,M)

        y_mu = jnp.mean(y_mu_mc, axis=1)
        # print(y_mu.shape, y_mu_mc.shape)

        # ===================
        # Covariance
        # ===================
        # (P,M) = (P,M) - (P, 1)
        dfydx = y_mu_mc - y_mu[:, None]

        # (P,D) = () * (P,M) @ (M,P)
        y_var = jnp.diag(self.wc * dfydx @ dfydx.T)

        y_var = jnp.atleast_1d(y_var)

        return y_mu, y_var

    def mean(self, f, x, x_cov):

        # sigma points
        sigma_pts = self.z.sample((self.n_samples,), self.rng_key)

        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)
        x_mc_samples = x[:, None] + L @ sigma_pts.T

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_mc = jax.vmap(f, in_axes=1, out_axes=1)(x_mc_samples)
        # print(y_mu_mc.shape, x_mc_samples.shape)

        # mean of mc samples
        # (P,) = (P,M)
        y_mu = jnp.mean(y_mu_mc, axis=1)
        # print(y_mu.shape, y_mu_mc.shape)

        return y_mu

    def covariance(self, f, x, x_cov):

        # sigma points
        sigma_pts = self.z.sample((self.n_samples,), self.rng_key)

        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_mc_samples = x[:, None] + L @ sigma_pts.T

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_mc = jax.vmap(f, in_axes=1, out_axes=1)(x_mc_samples)

        # mean of mc samples
        # (P,) = (P,M)
        y_mu = jnp.mean(y_mu_mc, axis=1)

        # ===================
        # Covariance
        # ===================
        # (P,M) = (P,M) - (P, 1)
        dfydx = y_mu_mc - y_mu[:, None]

        # (P,D) = () * (P,M) @ (M,P)
        y_cov = self.wc * dfydx @ dfydx.T

        return y_cov

    def variance(self, f, x, x_cov):

        y_cov = self.covariance(f, x, x_cov)

        y_var = jnp.diag(y_cov)

        y_var = jnp.atleast_1d(y_var)

        return y_var


def init_mc_moment_transform(
    meanf: Callable, n_features: int, mc_samples: int = 100, covariance: bool = False
) -> Callable:

    f = lambda x: jnp.atleast_1d(meanf(x).squeeze())
    z = dist.Normal(loc=jnp.zeros((n_features,)), scale=jnp.ones(n_features))
    _, wc = get_mc_weights(mc_samples)

    def apply_transform(rng_key, x, x_cov):

        # sigma points
        sigma_pts = z.sample((mc_samples,), rng_key)

        # cholesky for input covariance
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_mc_samples = x[:, None] + L @ sigma_pts.T
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_mc = jax.vmap(f, in_axes=1, out_axes=1)(x_mc_samples)

        # mean of mc samples
        # (P,) = (P,M)
        y_mu = jnp.mean(y_mu_mc, axis=1)

        # ===================
        # Covariance
        # ===================
        # (P,M) = (P,M) - (P, 1)
        dfydx = y_mu_mc - y_mu[:, None]

        # (P,D) = () * (P,M) @ (M,P)
        y_cov = wc * dfydx @ dfydx.T

        if not covariance:
            y_cov = jnp.diag(y_cov).reshape(-1)

        return y_mu, y_cov

    return apply_transform


# class MCTransform(objax.Module):
#     def __init__(self, model, jitted: bool = False, seed: int = 123):
#         self.n_features = model.X_train_.shape[-1]

#         self.key = jax.random.PRNGKey(seed)
#         f = jax.vmap(jax.partial(predictive_mean, model))

#         transform = jax.vmap(jax.partial(_transform, f))

#         if jitted:
#             transform = jax.jit(transform)

#         self.transform = transform

#     def forward(self, X, Xcov, n_samples: int) -> Tuple[JaxArray, JaxArray]:

#         self.wm, self.wc = get_mc_weights(n_samples)

#         key, self.key = jax.random.split(self.key, 2)
#         # generate mc samples
#         mc_samples = get_mc_sigma_points(
#             key, self.n_features, n_samples=(X.shape[0], n_samples)
#         )
#         # form sigma points from unit sigma-points
#         mean_f, var_f = self.transform(X, Xcov, mc_samples)

#         return mean_f, var_f


# def _transform(mean_f, X, Xcov, mc_samples):

#     # define the constants
#     wm = 1.0 / mc_samples.shape[0]
#     wc = 1.0 / (mc_samples.shape[0] - 1.0)

#     # form sigma points from unit sigma-points
#     x_ = X[:, None] + np.linalg.cholesky(Xcov) @ mc_samples.T

#     fx_ = mean_f(x_.T)

#     # output mean
#     mean_f = (wm * fx_).sum(axis=0)

#     # output covariance
#     dfx_ = (fx_ - mean_f)[:, None]

#     cov_f = wc * np.dot(dfx_.T, dfx_)

#     return mean_f, np.diag(cov_f)


def get_mc_weights(n_samples: int = 100) -> Tuple[float, float]:
    """Generate normalizers for MCMC samples"""
    mean = 1.0 / n_samples
    cov = 1.0 / (n_samples - 1)
    return mean, cov


def get_mc_sigma_points(rng_key, n_features: int, n_samples: int = 100) -> JaxArray:
    """Generate MCMC samples from a normal distribution"""

    sigma_dist = dist.Normal(loc=jnp.zeros((n_features,)), scale=jnp.ones(n_features))

    return sigma_dist.sample((n_samples,), rng_key)