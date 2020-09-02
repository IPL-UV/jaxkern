import jax
import jax.numpy as np

from src.kernels import gram
from src.kernels.utils import centering


def hsic(X, Y, kernel, params_x, params_y):

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


def centered_kernel_alignment(X, Y, kernel, params_x, params_y):

    # calculate hsic normally
    Pxy = hsic(X, Y, kernel, params_x, params_y)

    # calculate denominator
    Px = np.sqrt(hsic(X, X, kernel, params_x, params_x))
    Py = np.sqrt(hsic(Y, Y, kernel, params_y, params_y))

    cka_value = Pxy / (Px * Py)

    return cka_value


def kernel_alignment(X, Y, kernel, params_x, params_y):

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
