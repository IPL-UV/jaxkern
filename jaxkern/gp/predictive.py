from jaxkern.kernels.expectations import e_Kxy
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from objax.typing import JaxArray
from gpjax.gps import ConjugatePosterior
from chex import dataclass, Array
from gpjax.kernels import gram
from gpjax.utils import I
from jax.scipy.linalg import cho_factor, cho_solve
from gpjax.types import Dataset


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

    f = e_Kxy(gp.prior.kernel, param, mm_transform=mm_transform)

    mv = jax.vmap(f, in_axes=(0, None, None), out_axes=(0))
    psi1 = jax.vmap(mv, in_axes=(None, None, 0), out_axes=(1))

    def meanf(test_inputs: Array, test_cov: Array) -> Array:
        Kfx = psi1(test_inputs, test_cov, X)[..., 0]
        return jnp.dot(Kfx, weights)

    return meanf


def conditional(
    kernel: Callable,
    Xtrain: JaxArray,
    ytrain: JaxArray,
    weights: JaxArray,
    L: Tuple[JaxArray, bool],
    Xnew: JaxArray,
) -> Tuple[JaxArray, JaxArray]:
    # projection kernel
    K_Xx = kernel(Xnew, Xtrain)

    # Calculate the Mean
    mu_y = jnp.dot(K_Xx, weights)

    # calculate the covariance
    v = jax.scipy.linalg.cho_solve(L, K_Xx.T)

    K_xx = kernel(Xnew, Xnew)

    cov_y = K_xx - jnp.dot(K_Xx, v)

    return mu_y, cov_y


def conditional_mean(kernel, Xtrain, ytrain, weights, L, Xnew):
    return None


def predictive_mean(model, X: JaxArray) -> JaxArray:

    # projection kernel
    K_Xx = model.kernel(X, model.X_train_)

    # Calculate the Mean
    mu = jnp.dot(K_Xx, model.weights)

    return mu.squeeze()


def predictive_variance(model, X: JaxArray) -> JaxArray:

    # projection kernel
    K_Xx = model.kernel(X, model.X_train_)

    # Calculate the Mean
    v = jax.scipy.linalg.cho_solve(model.L, K_Xx.T)

    # calculate data kernel
    K_xx = model.kernel(X, X)

    cov_y = K_xx - jnp.dot(K_Xx, v)

    return jnp.diag(cov_y).squeeze()


def predictive_variance_y(model, X: JaxArray) -> JaxArray:

    # projection kernel
    K_Xx = model.kernel(X, model.X_train_)

    # Calculate the Mean
    v = jax.scipy.linalg.cho_solve(model.L, K_Xx.T)

    # calculate data kernel
    K_xx = model.kernel(X, X)

    cov_y = K_xx - jnp.dot(K_Xx, v)

    var_y = jnp.diag(cov_y).squeeze()

    var_y += jax.nn.softplus(model.noise.value) ** 2 * jnp.ones(cov_y.shape[0])

    return var_y
