from gpjax.gps import ConjugatePosterior
from chex import dataclass, Array
from gpjax.kernels import gram
from gpjax.utils import I
from jax.scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from jaxkern.kernels.expectations import Psi2, Psi0, Psi1
from jaxkern.gp.uncertain.moment import MomentTransform
from gpjax.types import Dataset
from jax import vmap
from typing import Callable
import jax.numpy as jnp


def moment_matching_predict_f(
    gp: ConjugatePosterior,
    param: dict,
    training: Dataset,
    mm_transform: dataclass,
    obs_noise: bool = True,
) -> Callable:
    X, y = training.X, training.y
    sigma = param["obs_noise"]
    n_train = training.n
    Kff = gram(gp.prior.kernel, X, param)
    L = cholesky(Kff + I(n_train) * sigma, lower=True)
    L_inv = solve_triangular(L.T, jnp.eye(L.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    prior_mean = gp.prior.mean_function(X)
    prior_distance = y - prior_mean
    weights = cho_solve((L, True), prior_distance)

    # ======================
    # Kernel Expectations
    # ======================
    # Psi0 - E_x[k(x,x)]
    psi0 = Psi0(gp.prior.kernel, param, mm_transform=mm_transform)

    # Psi1 - E_x[k(x,y)]
    psi1 = Psi1(gp.prior.kernel, param, mm_transform=mm_transform, Y=X)

    # Psi2 - E_x[k(x,y)k(x,z)]
    psi2 = Psi2(
        gp.prior.kernel,
        param,
        mm_transform=mm_transform,
        Y=X,
        Z=X,
    )

    def predict_f(test_inputs: Array, test_cov: Array) -> Array:

        # ====================
        # Mean Function
        # ====================

        Kfx = psi1(test_inputs, test_cov)

        y_mu = jnp.dot(Kfx.T, weights).squeeze()

        y_mu = jnp.atleast_1d(y_mu)

        # ====================
        # Variance Function
        # ====================

        # term 1
        t1 = psi0(test_inputs, test_cov)
        #         print(t1.shape)

        # term 2
        t2 = K_inv - weights @ weights.T
        #         print(t2.shape)
        t2 = t2 @ psi2(test_inputs, test_cov)
        #         print(t2.shape)
        t2 = jnp.trace(t2)

        # Term 3
        t3 = y_mu ** 2
        # #         print(t3.shape)
        # t3 = t3 @ t3.T @ weights @ weights.T
        # #         print(t3.shape)
        # t3 = jnp.trace(t3)
        # #         print(t3.shape)

        y_var = t1.squeeze() - t2.squeeze() - t3.squeeze()
        if obs_noise:
            y_var += sigma

        return jnp.atleast_1d(y_mu), jnp.atleast_1d(y_var)

    return predict_f


def moment_matching_mean(
    gp: ConjugatePosterior, param: dict, training: Dataset, mm_transform: dataclass
) -> Callable:
    X, y = training.X, training.y
    sigma = param["obs_noise"]
    n_train = training.n
    # Precompute covariance matrices
    Kff = gram(gp.prior.kernel, X, param)
    prior_mean = gp.prior.mean_function(X)
    L = cho_factor(Kff + I(n_train) * sigma, lower=True)

    prior_distance = y - prior_mean
    weights = cho_solve(L, prior_distance)

    psi1 = Psi1(gp.prior.kernel, param, mm_transform=mm_transform, Y=X)

    def meanf(test_inputs: Array, test_cov: Array) -> Array:
        Kfx = psi1(test_inputs, test_cov)

        y_mu = jnp.dot(Kfx.T, weights).squeeze()

        y_mu = jnp.atleast_1d(y_mu)

        return y_mu

    return meanf


def moment_matching_variance(
    gp: ConjugatePosterior,
    param: dict,
    training: Dataset,
    mm_transform: dataclass,
    obs_noise: bool = True,
) -> Callable:
    X, y = training.X, training.y
    sigma = param["obs_noise"]
    n_train = training.n
    Kff = gram(gp.prior.kernel, X, param)
    L = cholesky(Kff + I(n_train) * sigma, lower=True)
    L_inv = solve_triangular(L.T, jnp.eye(L.shape[0]))
    K_inv = L_inv.dot(L_inv.T)

    prior_mean = gp.prior.mean_function(X)
    prior_distance = y - prior_mean
    weights = cho_solve((L, True), prior_distance)

    if obs_noise:
        noise_constant = sigma
    else:
        noise_constant = 1.0

    # ======================
    # Kernel Expectations
    # ======================
    # Psi0 - E_x[k(x,x)]
    psi0 = Psi0(gp.prior.kernel, param, mm_transform=mm_transform)

    # Psi1 - E_x[k(x,y)]
    psi1 = Psi1(gp.prior.kernel, param, mm_transform=mm_transform, Y=X)

    # Psi2 - E_x[k(x,y)k(x,z)]
    psi2 = Psi2(
        gp.prior.kernel,
        param,
        mm_transform=mm_transform,
        Y=X,
        Z=X,
    )

    def varf(test_inputs: Array, test_cov: Array) -> Array:

        # term 1
        t1 = psi0(test_inputs, test_cov)
        #         print(t1.shape)

        # term 2
        t2 = K_inv - weights @ weights.T
        #         print(t2.shape)
        t2 = t2 @ psi2(test_inputs, test_cov)
        #         print(t2.shape)
        t2 = jnp.trace(t2)

        # Term 3
        t3 = psi1(test_inputs, test_cov)
        #         print(t3.shape)
        t3 = t3 @ t3.T @ weights @ weights.T
        #         print(t3.shape)
        t3 = jnp.trace(t3)
        #         print(t3.shape)

        t = t1.squeeze() - t2.squeeze() - t3.squeeze() + noise_constant

        t = jnp.atleast_1d(t)

        return t

    return varf
