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
import tqdm


from functools import partial


def iterate(data, n_layers=10):

    X_ldj = np.zeros(data.shape)
    forward_funcs = list()
    inverse_funcs = list()
    init_func = init_params_hist(10, 1_000, 1e-5)
    step = jax.jit(partial(get_rbig_params, init_func=init_func))
    with tqdm.trange(n_layers, desc="Layers") as pbar:
        for i in pbar:
            X_transform, iX_ldj, forward_func, inverse_func = step(data)
            # print(data.shape, X_transform.shape, X_ldj.shape, iX_ldj.shape)

            forward_funcs.append(forward_func)
            inverse_funcs.append(inverse_func)

            data = X_transform
            X_ldj += iX_ldj
    return data, X_ldj, forward_funcs, inverse_funcs


times = list()
dimensions = [2, 5, 10, 25, 50, 100, 1_000, 10_000]
n_layers = 100
n_samples = 10_000

with tqdm.tqdm(dimensions, desc="Dimensions") as pbar:
    for idim in pbar:
        fake_data = onp.random.randn(n_samples, idim)
        t0 = time.time()
        data, X_ldj, forward_funcs, inverse_funcs = iterate(fake_data, n_layers)
        times.append(time.time() - t0)


fig, ax = plt.subplots()
ax.plot(dimensions, times)
ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
plt.yscale("log")
plt.tight_layout()
fig.savefig("scripts/benchmark_dims.png")


times = list()
n_dimensions = 10
n_samples = [100, 1_000, 10_000, 100_000, 500_000, 1_000_000]
n_layers = 100

with tqdm.tqdm(n_samples, desc="Dimensions") as pbar:
    for i_samples in pbar:
        fake_data = onp.random.randn(i_samples, n_dimensions)
        t0 = time.time()
        data, X_ldj, forward_funcs, inverse_funcs = iterate(fake_data, n_layers)
        times.append(time.time() - t0)


fig, ax = plt.subplots()
ax.plot(n_samples, times)
ax.set(xlabel="Samples", ylabel="Timing (seconds)")
plt.yscale("log")
plt.tight_layout()
fig.savefig("scripts/benchmark_samples.png")
