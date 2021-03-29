import chex
import pytest
from jaxkern.gp.uncertain.mcmc import MCMomentTransform
from jaxkern.kernels.expectations import e_kx
from jaxkern.kernels.linear import linear_kernel
from jaxkern.kernels.stationary import rbf_kernel
import jax
import jax.random as jr
import jax.numpy as jnp

seed = 123
KEY = jr.PRNGKey(seed)


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("m_samples", [1, 5, 10])
def test_e_kx_shape_linear(n_features, m_samples):

    rng, x_rng = jr.split(KEY, 2)
    x = jr.normal(x_rng, shape=(n_features,))
    x_cov = jnp.ones(n_features)
    x_cov = jnp.diag(x_cov)

    chex.assert_shape(x, (n_features,))
    chex.assert_shape(x_cov, (n_features, n_features))

    # define kernel function
    kernel = lambda x, y: rbf_kernel(1.0, 1.0, x, y)
    rng, y_rng = jr.split(KEY, 2)
    Y = jr.normal(key=y_rng, shape=(m_samples, n_features))

    init_mcmc = MCMomentTransform(n_features=n_features, n_samples=10, seed=seed)

    # f = e_kx(wm, sigma_pts, kernel)
