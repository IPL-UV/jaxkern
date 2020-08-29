# JAX SETTINGS
import time as time
from functools import partial

import jax
import jax.numpy as np

# Plot Functions
import matplotlib.pyplot as plt
import numpy as onp
import seaborn as sns
import tqdm
from jax.config import config
from scipy.stats import beta

import wandb
from rbig_jax.data import get_classic
from rbig_jax.transforms.gaussian import get_gauss_params, init_params_hist
from rbig_jax.transforms.linear import init_pca_params
from rbig_jax.transforms.rbig import get_rbig_params

config.update("jax_enable_x64", True)


sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def main():
    wandb.init(project="rbigjax-benchmark", entity="emanjohnson91")
    wandb.config.dataset = "classic"
    wandb.config.implementation = "jax"
    wandb.config.support_extension = 10
    wandb.config.precision = 100
    wandb.config.alpha = 1e-5
    wandb.config.n_layers = 100
    # initialize parameters getter function
    init_func = init_params_hist(
        wandb.config.support_extension, wandb.config.precision, wandb.config.alpha,
    )

    # define step function (updates the parameters)
    step = jax.jit(partial(get_rbig_params, init_func=init_func))

    # ========================
    # Init Transformation
    # ========================

    def iterate(data, n_layers=10):

        # initialize the log determinant jacobian transforms
        X_ldj = np.zeros(data.shape)

        # prepare to grab the items
        forward_funcs = list()
        inverse_funcs = list()

        # loop through the number of layers
        with tqdm.trange(n_layers) as pbar:
            for i in pbar:

                # step through
                X_transform, iX_ldj, forward_func, inverse_func = step(data)

                # update data and ldj
                data = X_transform
                X_ldj += iX_ldj

        return data, X_ldj, forward_funcs, inverse_funcs

    times = list()
    dimensions = [2, 5, 10, 25, 50, 100, 1_000]
    n_layers = 100
    n_samples = 10_000

    with tqdm.tqdm(dimensions, desc="Dimensions") as pbar:
        for idim in pbar:
            fake_data = onp.random.randn(n_samples, idim)
            t0 = time.time()

            data, X_ldj, forward_funcs, inverse_funcs = iterate(fake_data, n_layers)

            wandb.log({"time": time.time() - t0}, step=idim)
            times.append(time.time() - t0)

    fig, ax = plt.subplots()
    ax.plot(dimensions, times)
    ax.set(xlabel="Dimensions", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    wandb.log({f"Dimensions": [wandb.Image(plt)]})

    times = list()
    n_dimensions = 10
    n_samples = [100, 1_000, 10_000, 100_000, 500_000, 1_000_000]
    n_layers = 100

    with tqdm.tqdm(n_samples, desc="Dimensions") as pbar:
        for i_samples in pbar:
            fake_data = onp.random.randn(i_samples, n_dimensions)
            t0 = time.time()
            data, X_ldj, forward_funcs, inverse_funcs = iterate(fake_data, n_layers)
            wandb.log({"time": time.time() - t0}, step=i_samples)
            times.append(time.time() - t0)

    fig, ax = plt.subplots()
    ax.plot(n_samples, times)
    ax.set(xlabel="Samples", ylabel="Timing (seconds)")
    plt.yscale("log")
    plt.tight_layout()
    wandb.log({f"Samples": [wandb.Image(plt)]})


if __name__ == "__main__":
    main()
