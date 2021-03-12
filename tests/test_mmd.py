from jaxkern.similarity.hsic import hsic_biased
import chex
import jax
import jax.numpy as np
from jax import random
import numpy as onp
import pytest
from jax import random

from jaxkern.similarity.hsic import hsic_biased
from jaxkern.similarity.mmd import mmd_mi, mmd_u_statistic, mmd_v_statistic
from jaxkern.kernels.kernels import ARDParams, ard_kernel

seed = 123
rng = onp.random.RandomState(123)
key = random.PRNGKey(123)


@pytest.mark.parametrize("n_samples", [10, 50, 100])
def test_mmd_u_stat_shape(n_samples):

    # generate random matrix
    K_x = random.normal(key, shape=(n_samples, n_samples))
    K_y = K_x @ random.uniform(key, shape=(n_samples, n_samples))
    K_xy = K_y @ random.uniform(key, shape=(n_samples, n_samples))

    # create
    score = mmd_u_statistic(K_x, K_y, K_xy)

    chex.assert_shape(score, ())


@pytest.mark.parametrize("n_samples", [10, 50, 100])
def test_mmd_v_stat_shape(n_samples):

    # generate random matrix
    K_x = random.normal(key, shape=(n_samples, n_samples))
    K_y = K_x @ random.uniform(key, shape=(n_samples, n_samples))
    K_xy = K_y @ random.uniform(key, shape=(n_samples, n_samples))

    # create
    score = mmd_v_statistic(K_x, K_y, K_xy)

    chex.assert_shape(score, ())


@pytest.mark.parametrize("n_samples", [10, 50, 100])
@pytest.mark.parametrize("n_features", [10, 50, 100])
def test_mmd_u_state_bounds(n_samples, n_features):

    # generate random matrix
    K_x = random.normal(key, shape=(n_samples, n_samples))
    K_y = K_x @ random.uniform(key, shape=(n_samples, n_samples))
    K_xy = K_y @ random.uniform(key, shape=(n_samples, n_samples))

    # create
    score = mmd_u_statistic(K_x, K_y, K_xy)

    assert score >= 0.0


@pytest.mark.parametrize("n_samples", [10, 50, 100])
@pytest.mark.parametrize("n_features", [10, 50, 100])
def test_mmd_v_state_bounds(n_samples, n_features):

    # generate random matrix
    K_x = random.normal(key, shape=(n_samples, n_samples))
    K_y = K_x @ random.uniform(key, shape=(n_samples, n_samples))
    K_xy = K_y @ random.uniform(key, shape=(n_samples, n_samples))

    # create
    score = mmd_v_statistic(K_x, K_y, K_xy)

    assert score >= 0.0


@pytest.mark.parametrize("n_samples", [10, 50, 100])
@pytest.mark.parametrize("n_features", [10, 50, 100])
def test_hsic_mmd_equal(n_samples, n_features):

    # generate random matrix
    X = random.normal(key, shape=(n_samples, n_features))
    Y = X @ random.uniform(key, shape=(n_features, n_features))

    # initialize parameters
    params = ARDParams(length_scale=1.0, variance=1.0)

    # calculate hsic statistic
    hsic = hsic_biased(ard_kernel, params, params, X=X, Y=Y)

    # calculate mmd statistic
    mmd = mmd_mi(ard_kernel, params, params, X=X, Y=Y)

    chex.assert_tree_all_close(hsic, mmd, rtol=1e-4)
