from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as np
from objax.typing import JaxArray


def cholesky_factorization(K: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Cholesky Factorization"""
    # cho factor the cholesky
    L = jax.scipy.linalg.cho_factor(K, lower=True)

    # weights
    weights = jax.scipy.linalg.cho_solve(L, Y)

    return L, weights


def saturate(params):
    """Softplus max on the params"""
    return {ikey: jax.nn.softplus(ivalue) for (ikey, ivalue) in params.items()}


def get_factorizations(
    X: np.ndarray,
    Y: np.ndarray,
    likelihood_noise: float,
    mean_f: Callable,
    kernel: Callable,
) -> Tuple[Tuple[np.ndarray, bool], np.ndarray]:
    """Cholesky Factorization"""

    # ==========================
    # 1. GP PRIOR
    # ==========================
    mu_x = mean_f(X)
    Kxx = kernel(X, X)

    # ===========================
    # 2. CHOLESKY FACTORIZATION
    # ===========================

    L, alpha = cholesky_factorization(
        Kxx + likelihood_noise ** 2 * np.eye(Kxx.shape[0]),
        Y.reshape(-1, 1) - mu_x.reshape(-1, 1),
    )

    # ================================
    # 4. PREDICTIVE MEAN DISTRIBUTION
    # ================================

    return L, alpha


def softplus_inverse(x: np.ndarray) -> np.ndarray:
    """Softplus inverse function

    Computes the element-wise function:

    .. math::
        \mathrm{softplus_inverse}(x) = \log(e^x - 1)
    """
    return np.log(np.exp(x) - 1.0)


def confidence_intervals(
    predictions: JaxArray, variance: JaxArray, ci: float = 96
) -> Tuple[JaxArray, JaxArray]:
    bound = (100 - ci) / 2
    ci_lower = predictions.squeeze() - bound * np.sqrt(variance.squeeze())
    ci_upper = predictions.squeeze() + bound * np.sqrt(variance.squeeze())
    return ci_lower, ci_upper
