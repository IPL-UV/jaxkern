from abc import abstractmethod
from typing import Callable, Tuple

import jax
from chex import Array
import jax.numpy as jnp

from jaxkern.gp.predictive import (
    predictive_mean,
    predictive_variance,
    predictive_variance_y,
)


def init_taylor_transform(
    meanf: Callable,
    varf: Callable,
) -> Callable:

    f = lambda x: meanf(x).squeeze()
    vf = lambda x: varf(x).squeeze()

    gradf = jax.grad(f)

    def apply_transform(x, x_cov):
        # ===================
        # Mean
        # ===================

        y_mu = taylor_o1_meanf(f, x)

        # ===================
        # Co/Variance
        # ===================
        y_var = taylor_o1_varf(gradf, vf, x, x_cov)

        return y_mu, y_var

    return apply_transform


def init_taylor_o2_transform(
    meanf: Callable,
    varf: Callable,
) -> Callable:

    # mean function
    f = lambda x: meanf(x).squeeze()
    # derivative of Mean function
    df = lambda x: jnp.atleast_1d(jax.grad(f)(x).squeeze())
    # 2nd derivative of mean function
    d2f = lambda x: jnp.atleast_2d(jax.hessian(f)(x).squeeze())
    # 2nd derivative of variance function
    vf = lambda x: jnp.atleast_1d(varf(x).squeeze())
    d2vf = jax.hessian(vf)

    def apply_transform(x, x_cov):
        # ===================
        # Mean
        # ===================

        y_mu = taylor_o2_meanf(f, d2f, x, x_cov)

        # ===================
        # Variance
        # ===================
        y_var = taylor_o2_varf(varf, d2vf, df, x, x_cov)

        return y_mu, y_var

    return apply_transform


def taylor_o1_meanf(f, x):
    y_mu = f(x)
    y_mu = jnp.atleast_1d(y_mu)
    return y_mu


def taylor_o2_meanf(f: Callable, d2f: Callable, x: Array, x_cov: Array) -> Array:
    y_mu = f(x)
    y_mu = jnp.atleast_1d(y_mu)
    d2y_mu = d2f(x)

    y_mu += 0.5 * jnp.trace(d2y_mu @ x_cov)

    return y_mu


def taylor_o1_varf(df, varf, x, x_cov):

    y_var = varf(x)
    y_var = jnp.atleast_1d(y_var)

    # gradient of the
    dy_mu = df(x)

    # gradient
    x_cov = jnp.atleast_2d(x_cov)

    y_var += dy_mu[None, :] @ x_cov @ dy_mu[:, None]

    return y_var[:, 0]


def taylor_o2_varf(
    varf: Callable, d2varf: Callable, df: Callable, x: Array, x_cov: Array
) -> Array:

    y_var = varf(x)
    y_var = jnp.atleast_1d(y_var)

    # 1st order correction
    # gradient of the
    dy_mu = df(x)

    # gradient
    x_cov = jnp.atleast_2d(x_cov)

    y_var += dy_mu[None, :] @ x_cov @ dy_mu[:, None]

    # 2nd order correction
    d2y_var = d2varf(x)
    y_var += 0.5 * jnp.trace(d2y_var @ x_cov)

    return y_var[:, 0]


#
# def taylor_first_order(
#     f: Callable, df: Callable, mean: JaxArray, cov: JaxArray
# ) -> Tuple[JaxArray, JaxArray]:
#     """Taylor First Order Transformation"""
#     # apply mean function
#     mean_f = f(mean)

#     # apply gradient mean function
#     jacobian_f = df(mean).reshape(1, -1)

#     # create covariance
#     cov_f = jacobian_f @ cov @ jacobian_f.T

#     return mean_f, np.diag(cov_f)


# def taylor_second_order(
#     f: Callable,
#     df: Callable,
#     d2f: Callable,
#     d2vf: Callable,
#     mean: JaxArray,
#     cov: JaxArray,
# ) -> Tuple[JaxArray, JaxArray]:
#     """Taylor Second Order Transformation"""
#     # apply 1st order taylor series
#     mean_f, var_f = taylor_first_order(f, df, mean, cov)

#     # apply mean correction
#     mu_df2 = d2f(mean)

#     mu_df2 = np.dot(mu_df2, cov)

#     mean_f += 0.5 * np.einsum("ii->", mu_df2)

#     # apply variance correction
#     var_df2 = d2vf(mean)

#     var_df2 = np.dot(var_df2, cov)

#     var_f += np.einsum("ii->", var_df2)

#     return mean_f, var_f


# class TaylorFirstOrder(objax.Module):
#     """Taylor First Order Transformation"""

#     def __init__(self, model, jitted: bool = True, noise: bool = True) -> None:
#         self.model = model
#         f = jax.partial(predictive_mean, model)
#         df = jax.grad(f)

#         # create transform
#         transform = jax.vmap(jax.partial(taylor_first_order, f, df), in_axes=(0, 0))

#         if jitted:
#             transform = jax.jit(transform)

#         if noise is True:
#             self.variance_f = predictive_variance_y
#         else:
#             self.variance_f = predictive_variance

#         self.transform = transform

#     def forward(self, X, Xcov) -> Tuple[JaxArray, JaxArray]:
#         # transformation
#         mu, var = self.transform(X, Xcov)

#         # add correction
#         var = self.variance_f(self.model, X).squeeze() + var.squeeze()

#         return mu, var


# class TaylorSecondOrder(objax.Module):
#     """Taylor Second Order Transformation"""

#     def __init__(self, model, jitted: bool = True, noise: bool = True) -> None:
#         self.model = model
#         f = jax.partial(predictive_mean, model)
#         df = jax.grad(f)
#         d2f = jax.hessian(f)
#         dvf2 = jax.hessian(jax.partial(predictive_variance, model))
#         transform = jax.vmap(
#             jax.partial(taylor_second_order, f, df, d2f, dvf2), in_axes=(0, 0)
#         )

#         if jitted:
#             transform = jax.jit(transform)

#         if noise:
#             self.variance_f = predictive_variance_y
#         else:
#             self.variance_f = predictive_variance

#         self.transform = transform

#     def forward(self, X, Xcov) -> Tuple[JaxArray, JaxArray]:
#         # transformation
#         mu, var = self.transform(X, Xcov)

#         # add correction
#         var = self.variance_f(self.model, X).squeeze() + var.squeeze()

#         return mu, var
