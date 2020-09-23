import jax.numpy as np
import numpy as onp
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances

from jaxkern.dist import distmat, pdist_squareform, sqeuclidean_distance

onp.random.seed(123)


def test_distmat():

    X = onp.random.rand(100, 2)

    dist = euclidean_distances(X, X, squared=True)
    dist_ = distmat(sqeuclidean_distance, X, X)
    onp.testing.assert_array_almost_equal(dist, onp.array(dist_))


def test_pdist_squareform():
    X = onp.random.randn(100, 2)

    dist = squareform(pdist(X, metric="sqeuclidean"))
    dist_ = pdist_squareform(X, X)
    onp.testing.assert_array_almost_equal(dist, onp.array(dist_), decimal=5)
