import jax.numpy as np
import numpy as onp
from sklearn.utils import check_random_state


def near_square_wave(
    n_train: int = 80,
    input_noise: float = 0.15,
    output_noise: float = 0.3,
    n_test: int = 400,
    random_state: int = 123,
):
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
    x_test = np.linspace(-12, 12, n_test) + x_rng.randn(n_test)
    y_test = f(np.linspace(-12, 12, n_test))
    x_test = np.sort(x_test)

    return x[:, None], y[:, None], x_test[:, None], y_test