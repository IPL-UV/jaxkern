import jax.numpy as np
import numpy as onp

from jaxkern.kernels import rbf_kernel, covariance_matrix, gram
from jaxkern.utils import centering
from sklearn.metrics.pairwise import rbf_kernel as rbf_sklearn
from sklearn.preprocessing import KernelCenterer

onp.random.seed(123)


def test_rbf_kernel_gram_1d():

    rng = onp.random.RandomState(123)
    n_samples = 100

    X = rng.rand(n_samples)

    # X
    K_sk = rbf_sklearn(X[:, np.newaxis], X[:, np.newaxis], gamma=1.0)

    K = gram(rbf_kernel, {"gamma": 1.0}, X, X)

    onp.testing.assert_array_almost_equal(K_sk, onp.array(K))

    Y = 10 * X + 0.1 * rng.randn(n_samples)

    # Y
    K_sk = rbf_sklearn(Y[:, np.newaxis], Y[:, np.newaxis], gamma=1.0)

    K = gram(rbf_kernel, {"gamma": 1.0}, Y, Y)

    onp.testing.assert_array_almost_equal(K_sk, onp.array(K))

    # X AND Y
    K_sk = rbf_sklearn(X[:, np.newaxis], Y[:, np.newaxis], gamma=1.0)

    K = gram(rbf_kernel, {"gamma": 1.0}, X, Y)

    onp.testing.assert_array_almost_equal(K_sk, onp.array(K))


def test_rbf_kernel_gram_2d():

    rng = onp.random.RandomState(123)
    n_samples, n_features = 100, 2
    X = onp.random.rand(n_samples, n_features)

    # sklearn rbf_kernel
    K_sk = rbf_sklearn(X, X, gamma=1.0)

    K = covariance_matrix(rbf_kernel, {"gamma": 1.0}, X, X)

    onp.testing.assert_array_almost_equal(K_sk, onp.array(K))

    Y = 10 * X + 0.1 * rng.randn(n_samples, n_features)

    # sklearn rbf_kernel
    K_sk = rbf_sklearn(Y, Y, gamma=1.0)

    K = gram(rbf_kernel, {"gamma": 1.0}, Y, Y)

    onp.testing.assert_array_almost_equal(K_sk, onp.array(K))


def test_rbf_kernel_cov_1d():

    X = onp.random.rand(100)

    # sklearn rbf_kernel
    K_sk = rbf_sklearn(X[:, np.newaxis], X[:, np.newaxis], gamma=1.0)

    K = covariance_matrix(rbf_kernel, {"gamma": 1.0}, X, X)

    onp.testing.assert_array_almost_equal(K_sk, onp.array(K))


def test_rbf_kernel_cov_2d():

    X = onp.random.rand(100, 2)

    # sklearn rbf_kernel
    K_sk = rbf_sklearn(X, X, gamma=1.0)

    K = gram(rbf_kernel, {"gamma": 1.0}, X, X)

    onp.testing.assert_array_almost_equal(K_sk, onp.array(K))


def test_centering():

    n_samples = 100

    X = onp.random.rand(n_samples)

    # sklearn rbf_kernel
    K_sk = rbf_sklearn(X[:, np.newaxis], X[:, np.newaxis], gamma=1.0)

    K_sk = KernelCenterer().fit_transform(K_sk)

    K = gram(rbf_kernel, {"gamma": 1.0}, X, X)
    # H = np.eye(n_samples) - (1.0 / n_samples) * np.ones((n_samples, n_samples))
    # K = np.einsum("ij,jk,kl->il", H, K, H)
    # K = np.dot(H, np.dot(K, H))
    K = centering(K)

    onp.testing.assert_array_almost_equal(K_sk, onp.array(K))