from abc import abstractmethod
from typing import Callable, Tuple

import jax
import jax.numpy as np
import objax
from objax.typing import JaxArray

from jaxkern.gp.predictive import (predictive_mean, predictive_variance,
                                   predictive_variance_y)


class TaylorFirstOrder(objax.Module):
    """Taylor First Order Transformation"""

    def __init__(self, model, jitted: bool = True, noise: bool = True) -> None:
        self.model = model
        f = jax.partial(predictive_mean, model)
        df = jax.grad(f)

        # create transform
        transform = jax.vmap(jax.partial(taylor_first_order, f, df), in_axes=(0, 0))

        if jitted:
            transform = jax.jit(transform)

        if noise is True:
            self.variance_f = predictive_variance_y
        else:
            self.variance_f = predictive_variance

        self.transform = transform

    def forward(self, X, Xcov) -> Tuple[JaxArray, JaxArray]:
        # transformation
        mu, var = self.transform(X, Xcov)

        # add correction
        var = self.variance_f(self.model, X).squeeze() + var.squeeze()

        return mu, var


class TaylorSecondOrder(objax.Module):
    """Taylor Second Order Transformation"""

    def __init__(self, model, jitted: bool = True, noise: bool = True) -> None:
        self.model = model
        f = jax.partial(predictive_mean, model)
        df = jax.grad(f)
        d2f = jax.hessian(f)
        dvf2 = jax.hessian(jax.partial(predictive_variance, model))
        transform = jax.vmap(
            jax.partial(taylor_second_order, f, df, d2f, dvf2), in_axes=(0, 0)
        )

        if jitted:
            transform = jax.jit(transform)

        if noise:
            self.variance_f = predictive_variance_y
        else:
            self.variance_f = predictive_variance

        self.transform = transform

    def forward(self, X, Xcov) -> Tuple[JaxArray, JaxArray]:
        # transformation
        mu, var = self.transform(X, Xcov)

        # add correction
        var = self.variance_f(self.model, X).squeeze() + var.squeeze()

        return mu, var


def taylor_first_order(
    f: Callable, df: Callable, mean: JaxArray, cov: JaxArray
) -> Tuple[JaxArray, JaxArray]:
    """Taylor First Order Transformation"""
    # apply mean function
    mean_f = f(mean)

    # apply gradient mean function
    jacobian_f = df(mean).reshape(1, -1)

    # create covariance
    cov_f = jacobian_f @ cov @ jacobian_f.T

    return mean_f, np.diag(cov_f)


def taylor_second_order(
    f: Callable,
    df: Callable,
    d2f: Callable,
    d2vf: Callable,
    mean: JaxArray,
    cov: JaxArray,
) -> Tuple[JaxArray, JaxArray]:
    """Taylor Second Order Transformation"""
    # apply 1st order taylor series
    mean_f, var_f = taylor_first_order(f, df, mean, cov)

    # apply mean correction
    mu_df2 = d2f(mean)

    mu_df2 = np.dot(mu_df2, cov)

    mean_f += 0.5 * np.einsum("ii->", mu_df2)

    # apply variance correction
    var_df2 = d2vf(mean)

    var_df2 = np.dot(var_df2, cov)

    var_f += np.einsum("ii->", var_df2)

    return mean_f, var_f
