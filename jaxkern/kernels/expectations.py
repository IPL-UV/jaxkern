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

    def __init__(
        self,
        kernel: Callable[[JaxArray, JaxArray], JaxArray],
        moment_transform: Callable,
        jitted: bool = False,
        **kwargs,
    ) -> None:

        moment_transform = moment_transform(**kwargs)
        sigma_points = moment_transform.sigma_points
        wm = moment_transform.wm

        transform_xkx = jax.partial(
            kernel_fx_expectation_vectorized,
            kernel,
            sigma_points,
            wm,
        )

        transform_xkxy = jax.partial(
            kernel_fxy_expectation_vectorized,
            kernel,
            sigma_points,
            wm,
        )
        transform_xkxyz = jax.partial(
            kernel_fxyz_expectation_vectorized,
            kernel,
            sigma_points,
            wm,
        )

        if jitted:
            transform_xkx = jax.jit(transform_xkx)
            transform_xkxy = jax.jit(transform_xkxy)
            transform_xkxyz = jax.jit(transform_xkxyz)

        self.moment_transform = moment_transform
        self.transform_xkx = transform_xkx
        self.transform_xkxy = transform_xkxy
        self.transform_xkxyz = transform_xkxyz
        self.kernel = kernel

    def expectation_xkx(self, X: JaxArray, Xcov: JaxArray) -> JaxArray:

        return self.transform_xkx(X, Xcov)

    def expectation_xkxy(
        self,
        X: JaxArray,
        Xcov: JaxArray,
        Y: JaxArray,
    ) -> JaxArray:
        return self.transform_xkxy(X, Xcov, Y)

    def expectation_xkxyz(
        self,
        X: JaxArray,
        Xcov: JaxArray,
        Y: JaxArray,
        Z: JaxArray,
    ) -> JaxArray:
        return self.transform_xkxyz(X, Xcov, Y, Z)


def mean_f_expectation(
    mean_f: Callable[[JaxArray], JaxArray],
    sigma_points: JaxArray,
    wm: JaxArray,
    X: JaxArray,
    Xcov: JaxArray,
) -> JaxArray:

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


def kernel_fx_expectation(
    kernel_f: Callable[[JaxArray], JaxArray],
    sigma_points: JaxArray,
    wm: JaxArray,
    X: JaxArray,
    Xcov: JaxArray,
) -> JaxArray:
    # get unit sigma points
    x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points

    # find values
    fx_ = jax.vmap(kernel_f)(x_.T, x_.T).squeeze()

    # output mean
    mean_f = np.sum(fx_ * wm)

    return mean_f


kernel_fx_expectation_vectorized = jax.vmap(
    kernel_fx_expectation, in_axes=(None, None, None, 0, 0)
)


def kernel_fxy_expectation(
    kernel_f: Callable[[JaxArray], JaxArray],
    sigma_points: JaxArray,
    wm: JaxArray,
    X: JaxArray,
    Xcov: JaxArray,
    Y: JaxArray,
) -> JaxArray:
    # get unit sigma points
    x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points

    # find values
    fx_ = jax.vmap(kernel_f, in_axes=(0, None))(x_.T, Y).squeeze()

    # output mean
    mean_f = np.sum(fx_ * wm)

    return mean_f


def kernel_fxy_expectation_vectorized(
    kernel_f: Callable[[JaxArray], JaxArray],
    sigma_points: JaxArray,
    wm: JaxArray,
    X: JaxArray,
    Xcov: JaxArray,
    Y: JaxArray,
):
    # do the cross vmap
    mv = jax.vmap(
        kernel_fxy_expectation, in_axes=(None, None, None, 0, 0, None), out_axes=0
    )
    mm = jax.vmap(mv, in_axes=(None, None, None, None, None, 0), out_axes=1)
    # calculate kernel
    return mm(kernel_f, sigma_points, wm, X, Xcov, Y)


def kernel_fxyz_expectation(
    kernel_f: Callable[[JaxArray], JaxArray],
    sigma_points: JaxArray,
    wm: JaxArray,
    X: JaxArray,
    Xcov: JaxArray,
    Z: JaxArray,
    Y: JaxArray,
) -> JaxArray:
    # get unit sigma points
    x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points

    def k(x, y, z):
        return kernel_f(y, x) * kernel_f(z, x)

    # find values
    fx_ = jax.vmap(k, in_axes=(0, None, None))(x_.T, Y, Z).squeeze()

    # output mean
    mean_f = np.sum(fx_ * wm)

    return mean_f


def kernel_fxyz_expectation_vectorized(
    kernel_f: Callable[[JaxArray], JaxArray],
    sigma_points: JaxArray,
    wm: JaxArray,
    X: JaxArray,
    Xcov: JaxArray,
    Y: JaxArray,
    Z: JaxArray,
):
    # do the cross vmap
    mvv = jax.vmap(
        kernel_fxyz_expectation,
        in_axes=(None, None, None, 0, 0, None, None),
        out_axes=0,
    )
    mmv = jax.vmap(mvv, in_axes=(None, None, None, None, None, 0, None), out_axes=1)
    mmm = jax.vmap(mmv, in_axes=(None, None, None, None, None, None, 0), out_axes=2)
    # calculate kernel
    return mmm(kernel_f, sigma_points, wm, X, Xcov, Y, Z)
