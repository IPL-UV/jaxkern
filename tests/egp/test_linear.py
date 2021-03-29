import chex
import pytest
from jaxkern.gp.uncertain.linear import init_taylor_transform, init_taylor_o2_transform
import jax
import jax.random as jr
import jax.numpy as jnp

seed = 123
KEY = jr.PRNGKey(seed)


@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_to1_shape(n_features):

    x = jr.normal(KEY, shape=(n_features,))
    x_cov = jnp.ones(n_features)
    x_cov = jnp.diag(x_cov)

    chex.assert_shape(x, (n_features,))
    chex.assert_shape(x_cov, (n_features, n_features))

    f = lambda x: jnp.sum(x) ** 2
    varf = lambda x: jnp.var(x)

    tayloro1_transform = init_taylor_transform(meanf=f, varf=varf)

    y_mu, y_var = tayloro1_transform(x, x_cov)

    chex.assert_equal_shape([y_mu, y_var])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_to1_vshape(n_samples, n_features):

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
    varf = lambda x: jnp.var(x)

    tayloro1_transform = init_taylor_transform(meanf=f, varf=varf)

    y_mu, y_var = jax.vmap(tayloro1_transform, in_axes=(0, None), out_axes=(0, 0))(
        x, x_cov
    )
    chex.assert_equal_shape([y_mu, y_var])
    chex.assert_shape(y_mu, (n_samples, 1))

    # # TODO: batch covariance for predictions?

    # tayloro1_transform = init_taylor_transform(meanf=f, covarf=covf, covariance=True)

    # y_mu, y_cov = jax.vmap(tayloro1_transform, in_axes=(0, None), out_axes=(0, 1))(
    #     x, x_cov
    # )

    # chex.assert_shape(y_mu, (n_samples, 1))
    # print(y_cov.shape, y_mu.shape)
    # chex.assert_shape(y_cov, (n_samples, n_samples, 1))


@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_to2_shape(n_features):

    x = jr.normal(KEY, shape=(n_features,))
    x_cov = jnp.ones(n_features)
    x_cov = jnp.diag(x_cov)

    chex.assert_shape(x, (n_features,))
    chex.assert_shape(x_cov, (n_features, n_features))

    f = lambda x: jnp.sum(x) ** 2
    varf = lambda x: jnp.var(x).squeeze()

    tayloro2_transform = init_taylor_o2_transform(meanf=f, varf=varf)

    y_mu, y_var = tayloro2_transform(x, x_cov)

    chex.assert_equal_shape([y_mu, y_var])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
def test_to2_vshape(n_samples, n_features):

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
    varf = lambda x: jnp.var(x).squeeze()

    tayloro2_transform = init_taylor_o2_transform(meanf=f, varf=varf)

    y_mu, y_var = jax.vmap(tayloro2_transform, in_axes=(0, None), out_axes=(0, 0))(
        x, x_cov
    )
    chex.assert_equal_shape([y_mu, y_var])
    chex.assert_shape(y_mu, (n_samples, 1))

    # # TODO: batch covariance for predictions?

    # tayloro1_transform = init_taylor_transform(meanf=f, covarf=covf, covariance=True)

    # y_mu, y_cov = jax.vmap(tayloro1_transform, in_axes=(0, None), out_axes=(0, 1))(
    #     x, x_cov
    # )

    # chex.assert_shape(y_mu, (n_samples, 1))
    # print(y_cov.shape, y_mu.shape)
    # chex.assert_shape(y_cov, (n_samples, n_samples, 1))
