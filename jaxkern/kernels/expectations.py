from typing import Callable

import jax
import jax.numpy as np
import objax
from objax.typing import JaxArray


class MeanExpectation(objax.Module):
    """Mean function expectations"""

    def __init__(
        self,
        mean: Callable[[JaxArray], JaxArray],
        moment_transform: Callable,
        jitted: bool = False,
        **kwargs,
    ):

        moment_transform = moment_transform(**kwargs)
        sigma_points = moment_transform.sigma_points
        wm = moment_transform.wm

        transform = jax.partial(
            mean_f_expectation_vectorized,
            mean,
            sigma_points,
            wm,
        )

        if jitted:
            transform = jax.jit(transform)

        self.mean = mean
        self.moment_transform = moment_transform
        self.transform = transform

    def e_px_mux(self, X: JaxArray, Xcov: JaxArray) -> JaxArray:

        return self.transform(X, Xcov)


class KernelExpectation(objax.Module):
    """Kernel Expectations"""

    def __init__(self, kernel: Callable[[JaxArray, JaxArray], JaxArray]) -> None:
        self.kernel = kernel

    def e_px_kxx(self, X: JaxArray, Xcov: JaxArray) -> JaxArray:
        pass

    def e_px_kxy(self, Y: JaxArray, X: JaxArray, Xcov: JaxArray) -> JaxArray:
        pass

    def e_px_kxykxz(
        self, Y: JaxArray, Z: JaxArray, X: JaxArray, Xcov: JaxArray
    ) -> JaxArray:
        pass


def mean_f_expectation(
    mean_f: Callable[[JaxArray], JaxArray],
    sigma_points: JaxArray,
    wm: JaxArray,
    X: JaxArray,
    Xcov: JaxArray,
) -> JaxArray:

    # form sigma points from unit sigma-points
    # print(Xcov.shape, Xcov.shape, sigma_points.shape)

    x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points
    # print("x_:", x_.shape)

    fx_ = mean_f(x_.T)
    # print("fx_:", fx_.shape, ", wm_:", wm.shape)

    # output mean
    mean_f = np.sum(fx_ * wm)

    return mean_f


mean_f_expectation_vectorized = jax.vmap(
    mean_f_expectation, in_axes=(None, None, None, 0, 0)
)
