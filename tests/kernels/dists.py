from src.kernels.dist import distmat, sqeuclidean_distance, pdist_squareform
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import numpy as onp
import jax.numpy as np

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

