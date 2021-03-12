import chex
import jax
import jax.numpy as np
from jax import random
import numpy as onp
import pytest
from jax import random

from jaxkern.similarity.hsic import (
    hsic_u_statistic_dot,
    hsic_v_statistic_trace,
    hsic_biased,
)
from jaxkern.kernels.kernels import ARDParams, ard_kernel
from jaxkern.kernels.utils import centering_matrix

seed = 123
rng = onp.random.RandomState(123)
key = random.PRNGKey(123)


@pytest.mark.parametrize("n_samples", [10, 50, 100])
def test_hsic_u_stat_shape(n_samples):

    # generate random matrix
    K_x = random.normal(key, shape=(n_samples, n_samples))
    K_y = random.uniform(key, shape=(n_samples, n_samples))

    # create
    hsic = hsic_u_statistic_dot(K_x, K_y)

    chex.assert_shape(hsic, ())


# @pytest.mark.parametrize("n_samples", [10, 50, 100])
# def test_hsic_u_stat_equal(n_samples):

#     # generate random matrix
#     K_x = random.normal(key, shape=(n_samples, n_samples))
#     K_y = random.uniform(key, shape=(n_samples, n_samples))

#     # create
#     hsic = hsic_u_statistic_dot(K_x, K_y)
#     hsic_einsum = hsic_u_statistic_einsum(K_x, K_y)

#     chex.assert_tree_all_close(hsic, hsic_einsum)


@pytest.mark.parametrize("n_samples", [10, 50, 100])
def test_hsic_v_stat_shape(n_samples):

    # generate random matrix
    K_x = random.normal(key, shape=(n_samples, n_samples))
    K_y = random.uniform(key, shape=(n_samples, n_samples))

    # create
    hsic = hsic_v_statistic_trace(K_x, K_y)

    chex.assert_shape(hsic, ())


# @pytest.mark.parametrize("n_samples", [10, 50, 100])
# def test_hsic_v_stat_equal(n_samples):

#     # generate random matrix
#     K_x = random.normal(key, shape=(n_samples, n_samples))
#     K_y = random.uniform(key, shape=(n_samples, n_samples)) @ K_x

#     # create
#     hsic_trace = hsic_v_statistic_trace(K_x, K_y)
#     hsic_einsum = hsic_v_statistic_einsum(K_x, K_y)

#     chex.assert_tree_all_close(hsic_trace, hsic_einsum)


@pytest.mark.parametrize("n_samples", [10, 50, 100])
@pytest.mark.parametrize("n_features", [10, 50, 100])
def test_hsic_shape(n_samples, n_features):

    # generate random matrix
    X = random.normal(key, shape=(n_samples, n_features))
    Y = X @ random.uniform(key, shape=(n_features, n_features))

    # initialize parameters
    params = ARDParams(length_scale=1.0, variance=1.0)

    # calculate hsic statistic
    hsic = hsic_biased(ard_kernel, params, params, X=X, Y=Y)

    chex.assert_shape(hsic, ())


@pytest.mark.parametrize("n_samples", [10, 50, 100])
@pytest.mark.parametrize("n_features", [10, 50, 100])
def test_hsic_bounds(n_samples, n_features):

    # generate random matrix
    X = random.normal(key, shape=(n_samples, n_features))
    Y = X @ random.uniform(key, shape=(n_features, n_features))

    # initialize parameters
    params = ARDParams(length_scale=1.0, variance=1.0)

    # calculate hsic statistic
    hsic = hsic_biased(ard_kernel, params, params, X=X, Y=Y)

    assert hsic >= 0.0
