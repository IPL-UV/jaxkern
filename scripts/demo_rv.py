import jax
import jax.numpy as np
import numpy as onp

from src.similarity.linear import rv_coeff, rv_coeff_features
from src.kernels import covariance_matrix, gram, rbf_kernel, linear_kernel
from src.kernels.dependence import centered_kernel_alignment, hsic, kernel_alignment
from src.kernels.utils import centering, gamma_from_sigma
from src.kernels.utils.sigma import estimate_sigma_median


def main():
    # generate some fake linear data
    X = onp.random.randn(1000, 2)
    Y = 2 * X + 0.05 * onp.random.randn(1000, 2)

    # t = centered_kernel_alignment(X, Y, linear_kernel, {}, {})
    rv_coeff_value = rv_coeff(X, Y)
    print(rv_coeff_value)

    # t = centered_kernel_alignment(X, Y, linear_kernel, {}, {})
    rv_coeff_value = rv_coeff_features(X, Y)
    print(rv_coeff_value)


if __name__ == "__main__":
    main()

