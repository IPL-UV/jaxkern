from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from chex import Array, dataclass
import jax.random as jr


@dataclass
class SigmaPointTransform:
    def predict_f(self, f: Callable, x: Array, x_cov: Array) -> Tuple[Array, Array]:

        # cholesky decomposition
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_sigma_samples = x[:, None] + L @ self.sigma_pts
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_sigma = jax.vmap(f, in_axes=1, out_axes=1)(x_sigma_samples)

        # mean of mc samples
        # (P,) = (P,M) @ (M,)
        y_mu = jnp.sum(y_mu_sigma * self.wm, axis=1)

        # ===================
        # Covariance
        # ===================
        # (P,M) = (P,M) - (P, 1)
        dfydx = y_mu_sigma - y_mu[:, None]

        # (P,D) = () * (P,M) @ (M,P)
        y_var = jnp.diag(dfydx @ self.wc @ dfydx.T)

        y_var = jnp.atleast_1d(y_var)

        return y_mu, y_var

    def mean(self, f: Callable, x: Array, x_cov: Array) -> Array:

        # cholesky decomposition
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_sigma_samples = x[:, None] + L @ self.sigma_pts
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_sigma = jax.vmap(f, in_axes=1, out_axes=1)(x_sigma_samples)

        # mean of mc samples
        # (P,) = (P,M) @ (M,)
        y_mu = y_mu_sigma @ self.wm

        return y_mu

    def covariance(self, f: Callable, x: Array, x_cov: Array) -> Array:

        # cholesky decomposition
        L = jnp.linalg.cholesky(x_cov)

        # calculate sigma points
        # (D,M) = (D,1) + (D,D)@(D,M)
        x_sigma_samples = x[:, None] + L @ self.sigma_pts
        # ===================
        # Mean
        # ===================

        # function predictions over mc samples
        # (P,M) = (D,M)
        y_mu_sigma = jax.vmap(f, in_axes=1, out_axes=1)(x_sigma_samples)

        # mean of mc samples
        # (P,) = (P,M) @ (M,)
        y_mu = y_mu_sigma @ self.wm

        # ===================
        # Covariance
        # ===================
        # (P,M) = (P,M) - (P, 1)
        dfydx = y_mu_sigma - y_mu[:, None]

        # (P,D) = () * (P,M) @ (M,P)
        y_cov = dfydx @ self.wc @ dfydx.T

        return y_cov

    def variance(self, f: Callable, x: Array, x_cov: Array) -> Array:

        y_cov = self.covariance(f, x, x_cov)

        y_var = jnp.diag(y_cov)

        y_var = jnp.atleast_1d(y_var)

        return y_var


def sigma_point_transform(
    sigma_pts: Array, Wm: Array, Wc: Array, covariance: bool, f, x: Array, x_cov: Array
):

    # cholesky decomposition
    L = jnp.linalg.cholesky(x_cov)

    # calculate sigma points
    # (D,M) = (D,1) + (D,D)@(D,M)
    x_mc_samples = x[:, None] + L @ sigma_pts
    # ===================
    # Mean
    # ===================

    # function predictions over mc samples
    # (P,M) = (D,M)
    y_mu_sigma = jax.vmap(f, in_axes=1, out_axes=1)(x_mc_samples)

    # mean of mc samples
    # (P,) = (P,M) @ (M,)
    y_mu = y_mu_sigma @ Wm

    # ===================
    # Covariance
    # ===================
    # (P,M) = (P,M) - (P, 1)
    dfydx = y_mu_sigma - y_mu[:, None]

    # (P,D) = (P, M) @ (M, M) @ (M, P)
    y_cov = dfydx @ Wc @ dfydx.T

    if not covariance:
        y_cov = jnp.diag(y_cov).reshape(-1)

    return y_mu, y_cov
