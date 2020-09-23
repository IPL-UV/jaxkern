import jax
import jax.numpy as np

from jaxkern.kernels import gram
from jaxkern.utils import centering

jax_np = jax.numpy.ndarray


def nhsic_cca(X, Y, kernel, params_x, params_y, epsilon=1e-5):

    n_samples = X.shape[0]

    # kernel matrix
    Kx = gram(kernel, params_x, X, X)
    Ky = gram(kernel, params_y, Y, Y)

    # center kernel matrices
    Kx = centering(Kx)
    Ky = centering(Ky)

    K_id = np.eye(Kx.shape[0])
    Kx_inv = np.linalg.inv(Kx + epsilon * n_samples * K_id)
    Ky_inv = np.linalg.inv(Ky + epsilon * n_samples * K_id)

    Rx = np.dot(Kx, Kx_inv)
    Ry = np.dot(Ky, Ky_inv)

    Pxy = np.mean(np.dot(Rx, Ry.T))

    return Pxy


def hsic(X, Y, kernel, params_x, params_y):
    """Calculates the HSIC similarity metric

    Parameters
    ----------
    X : jax.numpy.ndarray
        array-like of shape (n_samples, n_features)
    Y : np.ndarray
        The data matrix.
    """
    # kernel matrix
    Kx = gram(kernel, params_x, X, X)
    Ky = gram(kernel, params_y, Y, Y)

    # center kernel matrices
    Kx = centering(Kx)
    Ky = centering(Ky)

    #
    K = np.dot(Kx, Ky.T)

    hsic_value = np.mean(K)

    return hsic_value


def nhsic_cka(X, Y, kernel, params_x, params_y):

    # calculate hsic normally
    Pxy = hsic(X, Y, kernel, params_x, params_y)

    # calculate denominator
    Px = np.sqrt(hsic(X, X, kernel, params_x, params_x))
    Py = np.sqrt(hsic(Y, Y, kernel, params_y, params_y))

    cka_value = Pxy / (Px * Py)

    return cka_value


def nhsic_ka(X, Y, kernel, params_x, params_y):

    # calculate hsic normally
    Pxy = _hsic_uncentered(X, Y, kernel, params_x, params_y)

    # calculate denominator
    Px = np.sqrt(_hsic_uncentered(X, X, kernel, params_x, params_x))
    Py = np.sqrt(_hsic_uncentered(Y, Y, kernel, params_y, params_y))

    ka_value = Pxy / (Px * Py)

    return ka_value


def _hsic_uncentered(X, Y, kernel, params_x, params_y):

    # kernel matrix
    Kx = gram(kernel, params_x, X, X)
    Ky = gram(kernel, params_y, Y, Y)

    #
    K = np.dot(Kx, Ky.T)

    hsic_value = np.mean(K)

    return hsic_value
