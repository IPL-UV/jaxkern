import jax
import jax.numpy as np
import numpy as onp

from src.kernels import covariance_matrix, gram, rbf_kernel
from src.kernels.dependence import (centered_kernel_alignment, hsic,
                                    kernel_alignment)
from src.kernels.utils import centering, gamma_from_sigma
from src.kernels.utils.sigma import estimate_sigma_median

# generate some fake linear data
X = onp.random.randn(100, 2)
Y = 2 * X + 0.05 * onp.random.randn(100, 2)

# calculate the kernel matrix
sigma = estimate_sigma_median(X)  # estimate sigma value
params = {"gamma": gamma_from_sigma(sigma)}

# calculate hsic
hsic_value = hsic(X, Y, rbf_kernel, params, params)
print(hsic_value)


# derivative of hsic
dXhsic_value, dYhsic_value = jax.grad(hsic, argnums=(0, 1))(
    X, Y, rbf_kernel, params, params
)
print(dXhsic_value.shape, dYhsic_value.shape)

# calculate centered kernel alignment
cka_value = centered_kernel_alignment(X, Y, rbf_kernel, params, params)
print(cka_value)

# calculate kernel alignment
ka_value = kernel_alignment(X, Y, rbf_kernel, params, params)
print(ka_value)
