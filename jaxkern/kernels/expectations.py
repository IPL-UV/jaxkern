from typing import Callable, Dict

import jax
import jax.numpy as jnp
import objax
from chex import Array
from multipledispatch import dispatch


def e_Mx(mean: Callable, mm_transform: Callable) -> Callable:
    """kernel expectation, eMx

    Parameters
    ----------
    kernel : Callable,
        the kernel function be called

    Returns
    -------
    f : Callable
        Input: (n_features), (n_features, n_features)
        Output: (p_features)
    """

    def body(x):
        # calculate kernel
        y_mu = mean(x).squeeze()

        # ensure size
        y_mu = jnp.atleast_1d(y_mu)

        return y_mu

    def f(x, x_cov):

        x_mu = mm_transform.mean(body, x, x_cov)

        return x_mu

    return f


def e_Kx(kernel: Callable, params: Dict, mm_transform: Callable) -> Callable:
    """kernel expectation, eKdiag

    Parameters
    ----------
    wm : Array,
        the weights
    sigma_pts : Array
        the points to propagate through the function
    kernel : Callable,
        the kernel function be called

    Returns
    -------
    f : Callable
        Input: (n_features)
        Output: ()
    """

    def body(x):
        # calculate kernel
        kx = kernel(x, x, params)

        # ensure size
        kx = jnp.atleast_1d(kx)

        return kx

    def f(x, x_cov):

        x_mu = mm_transform.mean(body, x, x_cov)

        return x_mu

    return f


def e_Kxy(kernel: Callable, params: Dict, mm_transform: Callable) -> Callable:
    """kernel expectation, eKxy

    Parameters
    ----------
    kernel : Callable,
        the kernel function be called
    Y : Array
        the input data for the kernel, (m_features)

    Returns
    -------
    f : Callable
        Input: (n_features), (n_features, n_features)
        Output: (m_features)
    """

    def body(y, x):
        # calculate kernel
        kxy = kernel(x, y, params)

        # ensure size
        kxy = jnp.atleast_1d(kxy)

        return kxy

    def f(x, x_cov, y):

        f = jax.partial(body, y)

        x_mu = mm_transform.mean(f, x, x_cov)

        return jnp.atleast_1d(x_mu)

    return f


# def e_x_kxy(meanf: Callable, kernel: Callable, Y: Array) -> Callable:
#     raise NotImplementedError()


def e_Kxy_Kxz(
    kernel1: Callable,
    params1: Dict,
    kernel2: Callable,
    params2: Dict,
    mm_transform: Callable,
) -> Callable:
    """kernel expectation, eKxy

    Parameters
    ----------
    kernel_y : Callable,
        the kernel function be called for the input data, Y
    Y : Array
        the input data for the kernel, (m_features)
    kernel_z : Callable,
        the kernel function be called for the input data, Z
    Z : Array
        the input data for the kernel, (l_features)

    Returns
    -------
    f : Callable
        Input: (n_features), (n_features, n_features)
        Output: (m_features, l_features)
    """

    def body(y, z, x):
        # calculate kernel
        kxy = kernel1(x, y, params1)

        kxz = kernel2(x, z, params2)

        kxykxz = kxy * kxz

        # ensure size
        kxykxz = jnp.atleast_1d(kxykxz)

        return kxykxz

    def f(x, x_cov, y, z):

        f = jax.partial(body, y, z)

        x_mu = mm_transform.mean(f, x, x_cov)

        return jnp.atleast_1d(x_mu)

    return f


# class MeanExpectation(objax.Module):
#     """Mean function expectations"""

#     def __init__(
#         self,
#         mean: Callable[[Array], Array],
#         moment_transform: Callable,
#         jitted: bool = False,
#         **kwargs,
#     ):

#         moment_transform = moment_transform(**kwargs)
#         sigma_points = moment_transform.sigma_points
#         wm = moment_transform.wm

#         transform = jax.partial(
#             mean_f_expectation_vectorized,
#             mean,
#             sigma_points,
#             wm,
#         )

#         if jitted:
#             transform = jax.jit(transform)

#         self.mean = mean
#         self.moment_transform = moment_transform
#         self.transform = transform

#     def e_px_mux(self, X: Array, Xcov: Array) -> Array:

#         return self.transform(X, Xcov)


# class KernelExpectation(objax.Module):
#     """Kernel Expectations"""

#     def __init__(
#         self,
#         kernel: Callable[[Array, Array], Array],
#         moment_transform: Callable,
#         jitted: bool = False,
#         **kwargs,
#     ) -> None:

#         moment_transform = moment_transform(**kwargs)
#         sigma_points = moment_transform.sigma_points
#         wm = moment_transform.wm

#         transform_xkx = jax.partial(
#             kernel_fx_expectation_vectorized,
#             kernel,
#             sigma_points,
#             wm,
#         )

#         transform_xkxy = jax.partial(
#             kernel_fxy_expectation_vectorized,
#             kernel,
#             sigma_points,
#             wm,
#         )
#         transform_xkxyz = jax.partial(
#             kernel_fxyz_expectation_vectorized,
#             kernel,
#             sigma_points,
#             wm,
#         )

#         if jitted:
#             transform_xkx = jax.jit(transform_xkx)
#             transform_xkxy = jax.jit(transform_xkxy)
#             transform_xkxyz = jax.jit(transform_xkxyz)

#         self.moment_transform = moment_transform
#         self.transform_xkx = transform_xkx
#         self.transform_xkxy = transform_xkxy
#         self.transform_xkxyz = transform_xkxyz
#         self.kernel = kernel

#     def expectation_xkx(self, X: Array, Xcov: Array) -> Array:

#         return self.transform_xkx(X, Xcov)

#     def expectation_xkxy(
#         self,
#         X: Array,
#         Xcov: Array,
#         Y: Array,
#     ) -> Array:
#         return self.transform_xkxy(X, Xcov, Y)

#     def expectation_xkxyz(
#         self,
#         X: Array,
#         Xcov: Array,
#         Y: Array,
#         Z: Array,
#     ) -> Array:
#         return self.transform_xkxyz(X, Xcov, Y, Z)


# def mean_f_expectation(
#     mean_f: Callable[[Array], Array],
#     sigma_points: Array,
#     wm: Array,
#     X: Array,
#     Xcov: Array,
# ) -> Array:

#     x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points
#     # print("x_:", x_.shape)

#     fx_ = mean_f(x_.T)
#     # print("fx_:", fx_.shape, ", wm_:", wm.shape)

#     # output mean
#     mean_f = np.sum(fx_ * wm)

#     return mean_f


# mean_f_expectation_vectorized = jax.vmap(
#     mean_f_expectation, in_axes=(None, None, None, 0, 0)
# )


# def kernel_fx_expectation(
#     kernel_f: Callable[[Array], Array],
#     sigma_points: Array,
#     wm: Array,
#     X: Array,
#     Xcov: Array,
# ) -> Array:
#     # get unit sigma points
#     x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points

#     # find values
#     fx_ = jax.vmap(kernel_f)(x_.T, x_.T).squeeze()

#     # output mean
#     mean_f = np.sum(fx_ * wm)

#     return mean_f


# kernel_fx_expectation_vectorized = jax.vmap(
#     kernel_fx_expectation, in_axes=(None, None, None, 0, 0)
# )


# def kernel_fxy_expectation(
#     kernel_f: Callable[[Array], Array],
#     sigma_points: Array,
#     wm: Array,
#     X: Array,
#     Xcov: Array,
#     Y: Array,
# ) -> Array:
#     # get unit sigma points
#     x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points

#     # find values
#     fx_ = jax.vmap(kernel_f, in_axes=(0, None))(x_.T, Y).squeeze()

#     # output mean
#     mean_f = np.sum(fx_ * wm)

#     return mean_f


# def kernel_fxy_expectation_vectorized(
#     kernel_f: Callable[[Array], Array],
#     sigma_points: Array,
#     wm: Array,
#     X: Array,
#     Xcov: Array,
#     Y: Array,
# ):
#     # do the cross vmap
#     mv = jax.vmap(
#         kernel_fxy_expectation, in_axes=(None, None, None, 0, 0, None), out_axes=0
#     )
#     mm = jax.vmap(mv, in_axes=(None, None, None, None, None, 0), out_axes=1)
#     # calculate kernel
#     return mm(kernel_f, sigma_points, wm, X, Xcov, Y)


# def kernel_fxyz_expectation(
#     kernel_f: Callable[[Array], Array],
#     sigma_points: Array,
#     wm: Array,
#     X: Array,
#     Xcov: Array,
#     Z: Array,
#     Y: Array,
# ) -> Array:
#     # get unit sigma points
#     x_ = X[:, None] + np.linalg.cholesky(Xcov) @ sigma_points

#     def k(x, y, z):
#         return kernel_f(y, x) * kernel_f(z, x)

#     # find values
#     fx_ = jax.vmap(k, in_axes=(0, None, None))(x_.T, Y, Z).squeeze()

#     # output mean
#     mean_f = np.sum(fx_ * wm)

#     return mean_f


# def kernel_fxyz_expectation_vectorized(
#     kernel_f: Callable[[Array], Array],
#     sigma_points: Array,
#     wm: Array,
#     X: Array,
#     Xcov: Array,
#     Y: Array,
#     Z: Array,
# ):
#     # do the cross vmap
#     mvv = jax.vmap(
#         kernel_fxyz_expectation,
#         in_axes=(None, None, None, 0, 0, None, None),
#         out_axes=0,
#     )
#     mmv = jax.vmap(mvv, in_axes=(None, None, None, None, None, 0, None), out_axes=1)
#     mmm = jax.vmap(mmv, in_axes=(None, None, None, None, None, None, 0), out_axes=2)
#     # calculate kernel
#     return mmm(kernel_f, sigma_points, wm, X, Xcov, Y, Z)
