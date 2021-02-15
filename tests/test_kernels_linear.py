from sklearn.metrics.pairwise import (
    linear_kernel as linear_sklearn,
)
import jax.numpy as np
import numpy as onp
import chex
import pytest

from jaxkern.kernels.linear import Linear, linear_kernel

onp.random.seed(123)

rng = onp.random.RandomState(123)


def test_linear_shape():

    X = rng.randn(100, 2)

    linear_kern = Linear()

    K = linear_kern(X, X)

    assert K.shape == (100, 100)


def test_linear_f_shape():

    X = rng.randn(2, 2)

    k = linear_kernel(X[0], X[1])

    assert k.shape == ()


@pytest.mark.parametrize("X", [rng.randn(100, 1), rng.randn(100, 2), rng.randn(2, 10)])
def test_linear_result(X):

    linear_kern = Linear()

    K = linear_kern(X, X)
    K_sk = linear_sklearn(X, X)

    chex.assert_tree_all_close(K, np.array(K_sk), rtol=1e-4)


@pytest.mark.parametrize("X", [rng.randn(100, 1), rng.randn(100, 2), rng.randn(2, 10)])
def test_linear_f_result(X):

    X = rng.randn(2, 2)

    k = linear_kernel(X[0], X[1])

    k_sk = linear_sklearn(X[0][None, :], X[1][None, :])

    chex.assert_tree_all_close(k, np.array(k_sk), atol=1e-4)
