# JAX SETTINGS
import jax
import jax.numpy as np
import numpy as onp
from scipy.stats import beta

# Plot Functions
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

from rbig_jax.data import get_classic
from rbig_jax.transforms.gaussian import (
    init_params_hist,
    get_gauss_params,
    # forward_gaussianization,
    # inverse_gaussianization,
)
from rbig_jax.transforms.marginal import forward_gaussianization


# =========================
# Original Data
# =========================

data = get_classic(1_000).T

# ========================
# PLOT
# ========================
fig = plt.figure(figsize=(5, 5))

color = "blue"
title = "Original Data"
g = sns.jointplot(x=data[0], y=data[1], kind="hex", color=color)
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_x.png")

# ========================
# Forward Transformation
# ========================

# initialize parameters getter function
apply_func = init_params_hist(10, 1_000, 1e-5)

# get gaussian params
X_transform, Xldj, params = get_gauss_params(data, apply_func)

print(X_transform.shape)


color = "Red"
title = "Transformed Data"
g = sns.jointplot(x=X_transform[0], y=X_transform[1], kind="hex", color=color)
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_xg.png")

# ===========================
# Forward Transformation
# ===========================
X_transform, Xldj = jax.vmap(forward_gaussianization)(data, params)

color = "Red"
title = "Transformed Data"
g = sns.jointplot(x=X_transform[0], y=X_transform[1], kind="hex", color=color)
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_xg_forward.png")

# color = "Red"
# title = "Transformed Data"
# g = sns.jointplot(x=X_transform[:, 0], y=X_transform[:, 1], kind="hex", color=color)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.suptitle(title)
# plt.tight_layout()
# plt.savefig("scripts/demo2d_xu.png")

# color = "Red"
# title = "Transformed Data"
# g = sns.jointplot(x=X_transform[:, 0], y=X_transform[:, 1], kind="hex", color=color)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.suptitle(title)
# plt.tight_layout()
# plt.savefig("scripts/demo2d_d.png")


# # ===============================
# # Forward transformation
# # ===============================

# # initialize parameters getter function
# apply_func = init_params_hist_1d(10, 1_000, 1e-5)

# # get gaussian params
# X_g, Xldj, params = get_gauss_params(x_samples, apply_func)


# plt.figure()
# plt.hist(X_g, bins=100)
# plt.savefig("scripts/demo1d_xg.png")

# plt.figure()
# plt.hist(Xldj, bins=100)
# plt.savefig("scripts/demo1d_dx.png")

# # # Check forward transformation function
# X_g, Xldj = forward_gaussianization(x_samples, params)

# plt.figure()
# plt.hist(X_g, bins=100)
# plt.savefig("scripts/demo1d_xg_forward.png")

# # ===================================
# # Inverse Transformation
# # ===================================

# # Inverse Gaussian CDF (CDF function)
# X_approx = inverse_gaussianization(X_g, params)

# # # inverse uniformization (quantile function)
# # X_approx = inverse_uniformization_1d(X_u_approx, params)


# plt.figure()
# plt.hist(X_approx, bins=100)
# plt.savefig("scripts/demo1d_x_approx.png")


# # ===================================
# # Samples
# # ===================================


# # sample from a gaussian distribution
# X_g_samples = onp.random.randn(10_000)

# # Inverse Gaussian CDF (CDF function)
# X_samples = inverse_gaussianization(X_g_samples, params)

# # # inverse uniformization (quantile function)
# # X_approx = inverse_uniformization_1d(X_u_approx, params)


# plt.figure()
# plt.hist(X_samples, bins=100)
# plt.savefig("scripts/demo1d_x_samples.png")

# # ===================================
# # Probabilities
# # ===================================

# X_lprob = jax.scipy.stats.norm.logpdf(X_g) + Xldj

# plt.figure()
# plt.hist(X_lprob, bins=100)
# plt.savefig("scripts/demo1d_xlp_approx.png")

# plt.figure()
# plt.hist(data_dist.logpdf(x_samples), bins=100)
# plt.savefig("scripts/demo1d_xlp.png")

# print(X_lprob.sum())
# print(data_dist.logpdf(x_samples).sum())
