from scripts.demo_rv import main
import jax
import jax.numpy as np
import numpy as onp

from src.kernels import covariance_matrix, gram, rbf_kernel
from src.kernels.dependence import centered_kernel_alignment, hsic, kernel_alignment
from src.kernels.utils import centering, gamma_from_sigma
from src.kernels.utils.sigma import estimate_sigma_median


def main():
    # generate some fake linear data
    onp.random.seed(123)
    X = onp.random.randn(1000, 2)
    Y = 2 * X + 0.05 * onp.random.randn(1000, 2)

    # calculate the kernel matrix
    sigma = estimate_sigma_median(X)  # estimate sigma value
    params = {"gamma": gamma_from_sigma(sigma)}

    # calculate hsic

    hsic_value = hsic(X, Y, rbf_kernel, params, params)
    print(hsic_value)

    # calculate centered kernel alignment
    cka_value = centered_kernel_alignment(X, Y, rbf_kernel, params, params)
    print(cka_value)

    # calculate kernel alignment
    ka_value = kernel_alignment(X, Y, rbf_kernel, params, params)
    print(ka_value)


if __name__ == "__main__":
    main()
