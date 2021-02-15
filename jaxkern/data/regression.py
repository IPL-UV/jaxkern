from typing import Tuple

import jax.numpy as np
from objax.typing import JaxArray
import numpy as onp
from sklearn.utils import check_random_state


def simple(
    n_train: int = 30,
    n_test: int = 400,
    input_noise: float = 0.15,
    output_noise: float = 0.15,
    random_state: int = 123,
) -> Tuple[JaxArray, JaxArray, JaxArray, JaxArray]:
    """Generates a simple non-linear function"""
    onp.random.seed(random_state)
    X = np.linspace(-1, 1, n_train)

    def f(X):
        return (
            X + 0.2 * np.power(X, 3.0) + 0.5 * np.power(0.5 + X, 2.0) * np.sin(4.0 * X)
        )

    Y = f(X)
    Y += output_noise * onp.random.randn(n_train)
    Y -= np.mean(Y)
    Y /= np.std(Y)

    X += input_noise * onp.random.randn(n_train)

    assert X.shape == (n_train,)
    assert Y.shape == (n_train,)

    X_test = np.linspace(-1.2, 1.2, n_test)
    y_test = f(X_test)

    return X[:, None], Y[:, None], X_test[:, None], y_test[:, None]


def near_square_wave(
    n_train: int = 80,
    n_test: int = 400,
    input_noise: float = 0.3,
    output_noise: float = 0.05,
    random_state: int = 123,
) -> Tuple[JaxArray, JaxArray, JaxArray, JaxArray]:
    """Generates a near-square wave"""
    # function
    f = lambda x: np.sin(1.0 * np.pi / 1.6 * np.cos(5 + 0.5 * x))

    # create clean inputs
    x_mu = np.linspace(-10, 10, n_train)

    # clean outputs
    y = f(x_mu)

    # generate noise
    x_rng = check_random_state(random_state)
    y_rng = check_random_state(random_state + 1)

    # noisy inputs
    x = x_mu + input_noise * x_rng.randn(x_mu.shape[0])

    # noisy outputs
    y = f(x_mu) + output_noise * y_rng.randn(x_mu.shape[0])

    # test points
    x_test = np.linspace(-12, 12, n_test) + input_noise * x_rng.randn(n_test)
    y_test = f(np.linspace(-12, 12, n_test))
    x_test = np.sort(x_test)

    return x[:, None], y[:, None], x_test[:, None], y_test[:, None]
