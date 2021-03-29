import jax
import jax.numpy as jnp
from chex import Array


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
