import jax.numpy as np
import numpy as onp

# from jaxkern.kernels import rbf_kernel, covariance_matrix, gram
from jaxkern.utils import centering
from sklearn.metrics.pairwise import (
    rbf_kernel as rbf_sklearn,
    linear_kernel as linear_sklearn,
)
from sklearn.preprocessing import KernelCenterer

from jaxkern.kernels.stationary import Stationary

onp.random.seed(123)

rng = onp.random.RandomState(123)


def test_stationary_dist_shape():

    X = rng.randn(100, 2)

    stat_kern = Stationary()

    distmat = stat_kern.squared_distance(X, X)

    assert distmat.shape == (100, 100)


def test_stationary_diag_shape():

    X = rng.randn(100, 2)

    diag = Stationary().Kdiag(X)

    assert diag.shape == (100,)
