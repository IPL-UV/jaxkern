from typing import Callable, Tuple
import jax
import jax.numpy as np
from objax.typing import JaxArray


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
    mu_y = np.dot(K_Xx, weights)

    # calculate the covariance
    v = jax.scipy.linalg.cho_solve(L, K_Xx.T)

    K_xx = kernel(Xnew, Xnew)

    cov_y = K_xx - np.dot(K_Xx, v)

    return mu_y, cov_y


def conditional_mean(kernel, Xtrain, ytrain, weights, L, Xnew):
    return None


def predictive_mean(model, X: JaxArray) -> JaxArray:

    # projection kernel
    K_Xx = model.kernel(X, model.X_train_)

    # Calculate the Mean
    mu = np.dot(K_Xx, model.weights)

    return mu.squeeze()


def predictive_variance(model, X: JaxArray) -> JaxArray:

    # projection kernel
    K_Xx = model.kernel(X, model.X_train_)

    # Calculate the Mean
    v = jax.scipy.linalg.cho_solve(model.L, K_Xx.T)

    # calculate data kernel
    K_xx = model.kernel(X, X)

    cov_y = K_xx - np.dot(K_Xx, v)

    return np.diag(cov_y).squeeze()


def predictive_variance_y(model, X: JaxArray) -> JaxArray:

    # projection kernel
    K_Xx = model.kernel(X, model.X_train_)

    # Calculate the Mean
    v = jax.scipy.linalg.cho_solve(model.L, K_Xx.T)

    # calculate data kernel
    K_xx = model.kernel(X, X)

    cov_y = K_xx - np.dot(K_Xx, v)

    var_y = np.diag(cov_y).squeeze()

    var_y += jax.nn.softplus(model.noise.value) ** 2 * np.ones(cov_y.shape[0])

    return var_y
