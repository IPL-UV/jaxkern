import jax.numpy as np


def get_bin_edges(X: np.ndarray, nbins: int) -> np.ndarray:

    return np.linspace(np.min(X), np.max(X), nbins)


def compute_histogram(X: np.ndarray, nbins: int) -> np.ndarray:
    return None
