import jax.numpy as np

_float_eps = np.finfo("float").eps


def _ensure_min_eps(x):
    return np.maximum(_float_eps, x)


def centering(kernel_mat: np.ndarray) -> np.ndarray:
    """Calculates the centering matrix for the kernel"""
    n_samples = np.shape(kernel_mat)[0]

    identity = np.eye(n_samples)

    H = identity - (1.0 / n_samples) * np.ones((n_samples, n_samples))

    kernel_mat = np.dot(kernel_mat, H)

    return kernel_mat


def sigma_from_gamma(gamma=0.1):
    return _ensure_min_eps(np.sqrt(1.0 / (2 * gamma)))


def gamma_from_sigma(sigma=0.1):
    return _ensure_min_eps(1.0 / (2 * sigma ** 2))
