import jax
import jax.numpy as np
import numpy as onp

from jaxkern.similarity import HSIC, HSICRBF, CKARBF, CKA
from jaxkern.kernels.utils import centering, covariance_matrix, gram
from jaxkern.kernels.sigma import estimate_sigma_median


# generate some fake linear data
onp.random.seed(123)
X = onp.random.randn(1000, 2)
Y = 2 * X + 0.05 * onp.random.randn(1000, 2)


# def main():

#     # calculate the kernel matrix
#     sigma_x = estimate_sigma_median(X, X)  # estimate sigma value
#     params_x = {"gamma": gamma_from_sigma(sigma_x)}
#     sigma_y = estimate_sigma_median(Y, Y)  # estimate sigma value
#     params_y = {"gamma": gamma_from_sigma(sigma_y)}

#     # calculate hsic

#     hsic_value = hsic(X, Y, rbf_kernel, params_x, params_y)
#     print(f"HSIC: {hsic_value:.4f}")

#     # calculate centered kernel alignment
#     cka_value = nhsic_cka(X, Y, rbf_kernel, params_x, params_y)
#     print(f"nHSIC (CKA): {cka_value:.4f}")

#     nhsic_cca_value = nhsic_cca(X, Y, rbf_kernel, params_x, params_y)
#     print(f"nHSIC (CCA): {nhsic_cca_value:.4f}")
#     # calculate kernel alignment
#     ka_value = nhsic_ka(X, Y, rbf_kernel, params_x, params_y)
#     print(f"nHSIC (CCA): {ka_value:.4f}")


# if __name__ == "__main__":
#     main()
