from typing import Optional
import numpy as onp


def generate_data(
    n_samples: int = 1_000,
    dataset: str = "linear",
    noise_x: float = 0.1,
    noise_y: float = 0.1,
    random_state: Optional[int] = 123,
) -> None:

    rng = onp.random.RandomState(random_state)

    if dataset == "linear":
        x = rng.rand(n_samples, 1)

        y = x + noise_y * rng.randn(n_samples, 1)
    elif dataset == "random":
        x = rng.rand(n_samples, 1)
        y = rng.rand(n_samples, 1)

    elif dataset == "circle":
        t = 2 * onp.pi * rng.rand(n_samples, 1)
        x = onp.cos(t) + noise_x * rng.randn(n_samples, 1)
        y = onp.sin(t) + noise_y * rng.randn(n_samples, 1)

    else:
        raise ValueError(f"Unrecognized dataset: {dataset}")

    return x, y
