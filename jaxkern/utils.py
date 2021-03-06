import jax
import jax.numpy as np

_float_eps = np.finfo("float").eps


def ensure_min_eps(x: jax.numpy.ndarray) -> jax.numpy.ndarray:
    return np.maximum(_float_eps, x)


def centering(kernel_mat: jax.numpy.ndarray) -> jax.numpy.ndarray:
    """Calculates the centering matrix for the kernel"""
    n_samples = np.shape(kernel_mat)[0]

    identity = np.eye(n_samples)

    H = identity - (1.0 / n_samples) * np.ones((n_samples, n_samples))

    kernel_mat = np.einsum("ij,jk,kl->il", H, kernel_mat, H)

    return kernel_mat
