import jax
import jax.numpy as np
import numpy as onp

from jaxkern.similarity.linear import rv_coeff, rv_coeff_features


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

