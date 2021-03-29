import chex
import pytest
from jaxkern.gp.uncertain.unscented import (
    init_unscented_transform,
    get_unscented_sigma_points,
    get_unscented_weights,
)
import jax
import jax.random as jr
import jax.numpy as jnp

seed = 123
KEY = jr.PRNGKey(seed)


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize(
    "alpha",
    [
        1.0,
    ],
)
@pytest.mark.parametrize(
    "beta",
    [
        2.0,
    ],
)
@pytest.mark.parametrize(
    "kappa",
    [2.0, None],
)
def test_unscented_weights(n_features, alpha, beta, kappa):
    wm, wc = get_unscented_weights(n_features, kappa, alpha, beta)

    n_sigma_points = 2 * n_features + 1
    chex.assert_equal_shape([wm, wc])
    chex.assert_shape(wm, (n_sigma_points,))

    Wm, Wc = wm, jnp.diag(wc)

    chex.assert_shape(Wc, (n_sigma_points, n_sigma_points))
    chex.assert_shape(Wm, (n_sigma_points,))


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize(
    "alpha",
    [
        1.0,
    ],
)
@pytest.mark.parametrize(
    "kappa",
    [2.0, None],
)
def test_unscented_sigma_pts(n_features, alpha, kappa):
    sigma_pts = get_unscented_sigma_points(n_features, kappa, alpha)

    n_sigma_points = 2 * n_features + 1

    chex.assert_shape(sigma_pts, (n_features, n_sigma_points))


@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_unscented_transform_shape(n_features):

    alpha = 1.0
    beta = 2.0
    kappa = None

    x = jr.normal(KEY, shape=(n_features,))
    x_cov = jnp.ones(n_features)
    x_cov = jnp.diag(x_cov)

    chex.assert_shape(x, (n_features,))
    chex.assert_shape(x_cov, (n_features, n_features))

    f = lambda x: jnp.sum(x) ** 2

    unscented_transform = init_unscented_transform(
        meanf=f,
        n_features=n_features,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        covariance=False,
    )

    y_mu, y_var = unscented_transform(x, x_cov)

    chex.assert_equal_shape([y_mu, y_var])

    unscented_transform = init_unscented_transform(
        meanf=f,
        n_features=n_features,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        covariance=True,
    )

    y_mu, y_cov = unscented_transform(x, x_cov)

    chex.assert_shape(y_mu, (1,))
    chex.assert_shape(y_cov, (1, 1))


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_unscented_transform_vshape(n_samples, n_features):

    alpha = 1.0
    beta = 2.0
    kappa = None

    x = jr.normal(
        KEY,
        shape=(
            n_samples,
            n_features,
        ),
    )
    x_cov = jnp.ones(n_features)
    x_cov = jnp.diag(x_cov)

    chex.assert_shape(x, (n_samples, n_features))
    chex.assert_shape(x_cov, (n_features, n_features))

    f = lambda x: jnp.sum(x) ** 2

    unscented_transform = init_unscented_transform(
        meanf=f,
        n_features=n_features,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        covariance=False,
    )

    y_mu, y_var = jax.vmap(unscented_transform, in_axes=(0, None), out_axes=(0, 0))(
        x, x_cov
    )
    chex.assert_equal_shape([y_mu, y_var])
    chex.assert_shape(y_mu, (n_samples, 1))