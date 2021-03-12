import chex
import jax
import jax.numpy as np
from jax import random
import numpy as onp
import pytest
from jax import random

from jaxkern.similarity.cka import cka_biased
from jaxkern.kernels.kernels import ARDParams, ard_kernel
from jaxkern.kernels.utils import centering_matrix

seed = 123
rng = onp.random.RandomState(123)
key = random.PRNGKey(123)


@pytest.mark.parametrize("n_samples", [10, 50, 100])
@pytest.mark.parametrize("n_features", [10, 50, 100])
def test_cka_shape(n_samples, n_features):

    # generate random matrix
    X = random.normal(key, shape=(n_samples, n_features))
    Y = X @ random.uniform(key, shape=(n_features, n_features))

    # initialize parameters
    params = ARDParams(length_scale=1.0, variance=1.0)

    # calculate hsic statistic
    hsic = cka_biased(ard_kernel, params, params, X=X, Y=Y)

    chex.assert_shape(hsic, ())


@pytest.mark.parametrize("n_samples", [10, 50, 100])
@pytest.mark.parametrize("n_features", [10, 50, 100])
def test_cka_bounds(n_samples, n_features):

    # generate random matrix
    X = random.normal(key, shape=(n_samples, n_features))
    Y = X @ random.uniform(key, shape=(n_features, n_features))

    # initialize parameters
    params = ARDParams(length_scale=1.0, variance=1.0)

    # calculate hsic statistic
    cka = cka_biased(ard_kernel, params, params, X=X, Y=Y)

    assert cka >= 0.0
    assert cka <= 1.0