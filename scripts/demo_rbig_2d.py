# JAX SETTINGS
import jax
import jax.numpy as np
from jax.config import config

config.update("jax_enable_x64", True)

import numpy as onp
from scipy.stats import beta

import time as time

# Plot Functions
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)

from rbig_jax.data import get_classic
from rbig_jax.transforms.gaussian import (
    init_params_hist,
    get_gauss_params,
)
from rbig_jax.transforms.linear import init_pca_params
from rbig_jax.transforms.rbig import get_rbig_params
from rbig_jax.transforms.gaussian import init_params_hist
from rbig_jax.information.entropy import entropy_marginal
from rbig_jax.information.reduction import information_reduction
import tqdm

# =========================
# Original Data
# =========================

data = get_classic(10_000).T

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
plt.savefig("scripts/demo2d_rbig_x.png")


# ========================
# Init Transformation
# ========================

# initialize parameters getter function
init_func = init_params_hist(10, 1_000, 1e-5)

# # initialize parameters getter function
# apply_func = init_params_hist(10, 1_000, 1e-5)

# print(data.shape)
# # get gaussian params
# X_transform, Xldj, mg_params, forward_mg_func, inverse_mg_func = get_gauss_params(
#     data, apply_func
# )

# # get rotation params
# rot_params, forward_rot_func, inverse_rot_func = init_pca_params(X_transform.T)

# # forward transformation
# X_transform, X_ldj = forward_rot_func(X_transform.T)

# X_transform, X_ldj, forward_func, inverse_func = get_rbig_params(data.T, init_func)


# color = "Red"
# title = "Transformed Data"
# g = sns.jointplot(x=X_transform.T[0], y=X_transform.T[1], kind="hex", color=color)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.suptitle(title)
# plt.tight_layout()
# plt.savefig("scripts/demo2d_rbig_xg.png")

# # ========================
# # Forward Transformation
# # ========================
# t0 = time.time()
# X_transform, X_ldj = forward_func(data.T)
# print(f"Time: {time.time()-t0:.5f}")
# t0 = time.time()
# X_transform, X_ldj = forward_func(data.T)
# print(f"Time: {time.time()-t0:.5f}")

# color = "Red"
# title = "Transformed Data"
# g = sns.jointplot(x=X_transform.T[0], y=X_transform.T[1], kind="hex", color=color)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.suptitle(title)
# plt.tight_layout()
# plt.savefig("scripts/demo2d_rbig_xg_forward.png")


# # ========================
# # Inverse Transformation
# # ========================

# t0 = time.time()
# X_approx = inverse_func(X_transform)
# print(f"Time: {time.time()-t0:.5f}")
# t0 = time.time()
# X_approx = inverse_func(X_transform)
# print(f"Time: {time.time()-t0:.5f}")

# color = "Red"
# title = "Transformed Data"
# g = sns.jointplot(x=X_approx[0], y=X_approx[1], kind="hex", color=color)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.suptitle(title)
# plt.tight_layout()
# plt.savefig("scripts/demo2d_rbig_x_approx.png")
from functools import partial


def iterate(data, n_layers=10):

    X_ldj = np.zeros(data.shape)
    forward_funcs = list()
    inverse_funcs = list()
    it_red = list()
    step = jax.jit(partial(get_rbig_params, init_func=init_func))
    with tqdm.trange(n_layers) as pbar:
        for i in pbar:
            X_transform, iX_ldj, forward_func, inverse_func = step(data)
            # print(data.shape, X_transform.shape, X_ldj.shape, iX_ldj.shape)

            # h = jax.vmap(entropy_marginal)(X_transform.T)
            # print(h.shape, print(h))
            it = information_reduction(data, X_transform)
            # print(it)
            it_red.append(it)

            forward_funcs.append(forward_func)
            inverse_funcs.append(inverse_func)

            data = X_transform
            X_ldj += iX_ldj
    return data, X_ldj, forward_funcs, inverse_funcs, it_red


data, X_ldj, forward_funcs, inverse_funcs, it_red = iterate(data.T, 50)

color = "Red"
title = "Transformed Data"
g = sns.jointplot(x=data.T[0], y=data.T[1], kind="hex", color=color)
plt.xlabel("X")
plt.ylabel("Y")
plt.suptitle(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_rbig_xg_forward.png")

color = "Red"
title = f"Information Reduction, TC:{np.sum(it_red):.4f}"
plt.figure()
plt.plot(np.cumsum(np.array(it_red)))
plt.xlabel("Layers")
plt.ylabel("Total Correlation")
plt.title(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_rbig_xg_tc.png")

# color = "Red"
# title = "Transformed Data"
# g = sns.jointplot(
#     x=X_ldj.sum(axis=1).T[0], y=X_ldj.sum(axis=1).T[1], kind="hex", color=color
# )
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.suptitle(title)
# plt.tight_layout()
# plt.savefig("scripts/demo2d_rbig_dx_forward.png")

fake_data = onp.random.randn(10_000, 10)

data, X_ldj, forward_funcs, inverse_funcs, it_red = iterate(fake_data, 50)

color = "Red"
title = f"Information Reduction, TC:{np.sum(it_red):.4f}"
plt.figure()
plt.plot(np.cumsum(np.array(it_red)))
plt.xlabel("Layers")
plt.ylabel("Total Correlation")
plt.title(title)
plt.tight_layout()
plt.savefig("scripts/demo2d_rbig_xg_tc_2.png")
# def iterate(data, n_layers=10):

#     X_ldj = np.zeros(data.shape)
#     forward_funcs = list()
#     inverse_funcs = list()
#     step = jax.jit(partial(get_rbig_params, init_func=init_func))
#     with tqdm.trange(n_layers) as pbar:
#         for i in pbar:
#             data, iX_ldj, forward_func, inverse_func = step(data)
#             # print(data.shape, X_transform.shape, X_ldj.shape, iX_ldj.shape)

#             forward_funcs.append(forward_func)
#             inverse_funcs.append(inverse_func)

#             X_ldj = X_ldj + iX_ldj.T
#     return data, X_ldj, forward_funcs, inverse_funcs

