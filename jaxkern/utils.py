import jax
import jax.numpy as np
from objax.typing import JaxArray

_float_eps = np.finfo("float").eps


def ensure_min_eps(x: JaxArray) -> JaxArray:
    """Ensures no overflow or round-off errors"""
    return np.maximum(_float_eps, x)
