import chex
import pytest
from jaxkern.gp.uncertain.mcmc import init_mc_moment_transform
import jax
import jax.random as jr
import jax.numpy as jnp

seed = 123
KEY = jr.PRNGKey(seed)


@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_mcmc_shape(n_features):

    x = jr.normal(KEY, shape=(n_features,))
    x_cov = jnp.ones(n_features)
    x_cov = jnp.diag(x_cov)

    chex.assert_shape(x, (n_features,))
    chex.assert_shape(x_cov, (n_features, n_features))

    f = lambda x: jnp.sum(x) ** 2

    mc_transform = init_mc_moment_transform(
        meanf=f, n_features=n_features, mc_samples=100, covariance=False
    )

    y_mu, y_var = mc_transform(KEY, x, x_cov)

    chex.assert_equal_shape([y_mu, y_var])

    mc_transform = init_mc_moment_transform(
        meanf=f, n_features=n_features, mc_samples=100, covariance=True
    )

    y_mu, y_cov = mc_transform(KEY, x, x_cov)

    chex.assert_shape(y_mu, (1,))
    chex.assert_shape(y_cov, (1, 1))


@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_mcmc_mo_shape(n_features):

    x = jr.normal(KEY, shape=(n_features,))
    x_cov = jnp.ones(n_features)
    x_cov = jnp.diag(x_cov)

    chex.assert_shape(x, (n_features,))
    chex.assert_shape(x_cov, (n_features, n_features))

    f = lambda x: x ** 2

    mc_transform = init_mc_moment_transform(
        meanf=f, n_features=n_features, mc_samples=100, covariance=False
    )

    y_mu, y_var = mc_transform(KEY, x, x_cov)

    chex.assert_equal_shape([x, y_mu, y_var])

    mc_transform = init_mc_moment_transform(
        meanf=f, n_features=n_features, mc_samples=100, covariance=True
    )

    y_mu, y_cov = mc_transform(KEY, x, x_cov)

    chex.assert_equal_shape([x, y_mu])
    chex.assert_shape(y_cov, (n_features, n_features))


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_mcmc_vshape(n_samples, n_features):

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

    mc_transform = init_mc_moment_transform(
        meanf=f, n_features=n_features, mc_samples=100, covariance=False
    )

    y_mu, y_var = jax.vmap(mc_transform, in_axes=(None, 0, None), out_axes=(0, 0))(
        KEY, x, x_cov
    )
    chex.assert_equal_shape([y_mu, y_var])
    chex.assert_shape(y_mu, (n_samples, 1))

    # TODO: batch covariance for predictions?

    # mc_transform = init_mc_moment_transform(
    #     meanf=f, n_features=n_features, mc_samples=100, covariance=True
    # )

    # y_mu, y_cov = jax.vmap(mc_transform, in_axes=(None, 0, None), out_axes=(0, 1))(
    #     KEY, x, x_cov
    # )

    # chex.assert_shape(y_mu, (n_samples, 1))
    # print(y_cov.shape, y_mu.shape)
    # chex.assert_shape(y_cov, (n_samples, n_samples, 1))
