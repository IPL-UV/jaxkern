import jax
import jax.numpy as np
import numpy as onp

from jaxkern.kernels.utils.sigma import (
    estimate_sigma_median,
    estimate_sigma_median_kth,
    scotts_factor,
    silvermans_factor,
)


def main():
    # generate some fake linear data
    onp.random.seed(123)
    X = onp.random.randn(1000, 2)

    # calculate the kernel matrix
    sigma = estimate_sigma_median(X, X)  # estimate sigma value
    print(f"Median: {sigma:.4f}")
    # calculate the kernel matrix
    percent = 0.4
    sigma = estimate_sigma_median_kth(X, X, percent)  # estimate sigma value
    print(f"Median (percent={percent:.1f}): {sigma:.4f}")

    # Scotts Method
    sigma = scotts_factor(X)  # estimate sigma value
    print(f"Scott: {sigma:.4f}")

    # Silvermans method
    sigma = silvermans_factor(X)  # estimate sigma value
    print(f"Silverman: {sigma:.4f}")


if __name__ == "__main__":
    main()
