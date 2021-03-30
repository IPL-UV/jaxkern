from typing import Optional
from jaxkern.gp.uncertain.sigma import SigmaPointTransform
from numpy.polynomial.hermite_e import hermegauss, hermeval
from chex import Array, dataclass
from scipy.special import factorial
from sklearn.utils.extmath import cartesian
import numpy as np
import jax.numpy as jnp


@dataclass
class GaussHermite(SigmaPointTransform):
    n_features: int
    degree: int = 20

    def __post_init__(
        self,
    ) -> None:

        # generate sigma weights
        wm = get_quadrature_weights(n_features=self.n_features, degree=self.degree)
        self.wm = jnp.array(wm)
        self.wc = jnp.diag(wm)

        # generate sigma points
        sigma_pts = get_quadrature_sigma_points(
            n_features=self.n_features, degree=self.degree
        )
        self.sigma_pts = jnp.array(sigma_pts)


def get_quadrature_weights(
    n_features: int,
    degree: int = 3,
) -> Array:
    """Generate normalizers for MCMC samples"""

    # 1D sigma-points (x) and weights (w)
    x, w = hermegauss(degree)
    # hermegauss() provides weights that cause posdef errors
    w = factorial(degree) / (degree ** 2 * hermeval(x, [0] * (degree - 1) + [1]) ** 2)
    return np.prod(cartesian([w] * n_features), axis=1)


def get_quadrature_sigma_points(
    n_features: int,
    degree: int = 3,
) -> Array:
    """Generate Unscented samples"""
    # 1D sigma-points (x) and weights (w)
    x, w = hermegauss(degree)
    # nD sigma-points by cartesian product
    return cartesian([x] * n_features).T  # column/sigma-point