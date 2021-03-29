import chex
import jax
import jax.numpy as jnp
from typing import Callable, Optional, Tuple
from jaxkern.gp.uncertain.sigma import sigma_point_transform


def init_unscented_transform(
    meanf: Callable,
    n_features: int,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: Optional[float] = None,
    covariance: bool = False,
) -> Callable:

    f = lambda x: jnp.atleast_1d(meanf(x).squeeze())

    # get weights
    wm, wc = get_unscented_weights(n_features, kappa, alpha, beta)
    Wm, Wc = wm, jnp.diag(wc)

    # generate sigma points
    sigma_pts = get_unscented_sigma_points(n_features, kappa, alpha)

    apply_transform = jax.partial(
        sigma_point_transform, sigma_pts, Wm, Wc, covariance, f
    )

    return apply_transform


def get_unscented_sigma_points(
    n_features: int, kappa: Optional[float] = None, alpha: float = 1.0
) -> Tuple[chex.Array, chex.Array]:
    """Generate Unscented samples"""

    # calculate kappa value
    if kappa is None:
        kappa = jnp.maximum(3.0 - n_features, 0.0)

    lam = alpha ** 2 * (n_features + kappa) - n_features
    c = jnp.sqrt(n_features + lam)
    return jnp.hstack(
        (jnp.zeros((n_features, 1)), c * jnp.eye(n_features), -c * jnp.eye(n_features))
    )


def get_unscented_weights(
    n_features: int,
    kappa: Optional[float] = None,
    alpha: float = 1.0,
    beta: float = 2.0,
) -> Tuple[float, float]:
    """Generate normalizers for MCMC samples"""

    # calculate kappa value
    if kappa is None:
        kappa = jnp.maximum(3.0 - n_features, 0.0)

    lam = alpha ** 2 * (n_features + kappa) - n_features
    wm = 1.0 / (2.0 * (n_features + lam)) * jnp.ones(2 * n_features + 1)
    wc = wm.copy()
    wm = jax.ops.index_update(wm, 0, lam / (n_features + lam))
    wc = jax.ops.index_update(wc, 0, wm[0] + (1 - alpha ** 2 + beta))
    return wm, wc
