import time as time
from functools import partial

import jax
import jax.numpy as np
import numpy as onp
import seaborn as sns
import tqdm
from jax.config import config

import wandb
from rbig_jax.data import get_classic
from rbig_jax.information.reduction import information_reduction
from rbig_jax.plots.info import plot_total_corr
from rbig_jax.plots.joint import plot_joint
from rbig_jax.transforms.gaussian import init_params_hist
from rbig_jax.transforms.rbig import get_rbig_params

config.update("jax_enable_x64", True)


sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)


def main():
    # =========================
    # Logger
    # =========================
    wandb.init(project="rbigjax-demo-2d", entity="emanjohnson91")

    # config parameters
    wandb.config.n_samples = 10_000
    wandb.config.dataset = "classic"
    wandb.config.support_extension = 10
    wandb.config.precision = 100
    wandb.config.alpha = 1e-5
    wandb.config.n_layers = 100

    # =========================
    # Original Data
    # =========================

    data = get_classic(wandb.config.n_samples).T

    # ========================
    # PLOT
    # ========================
    plot_joint(data.T, "blue", "Original Data", True)

    # ========================
    # Init Transformation
    # ========================

    # initialize parameters getter function
    init_func = init_params_hist(
        wandb.config.support_extension, wandb.config.precision, wandb.config.alpha,
    )

    # define step function (updates the parameters)
    step = jax.jit(partial(get_rbig_params, init_func=init_func))

    def iterate(data, n_layers=10):

        # initialize the log determinant jacobian transforms
        X_ldj = np.zeros(data.shape)

        # prepare to grab the items
        forward_funcs = list()
        inverse_funcs = list()
        it_red = list()

        # loop through the number of layers
        with tqdm.trange(n_layers) as pbar:
            for i in pbar:

                # step through
                X_transform, iX_ldj, forward_func, inverse_func = step(data)

                # calculate the information loss
                it = information_reduction(data, X_transform)
                wandb.log({"Delta Multi-Information": onp.array(it)})

                # calculate the running total corrlation
                it_red.append(it)
                tc = np.array(it_red).sum()
                wandb.log({"TC": onp.array(tc)})

                # save functions
                forward_funcs.append(forward_func)
                inverse_funcs.append(inverse_func)

                # update data and ldj
                data = X_transform
                X_ldj += iX_ldj

                # calculate negative log likelihood
                nll = jax.scipy.stats.norm.logpdf(data) + X_ldj
                nll = nll.sum()
                wandb.log({"Negative Log-Likelihood": onp.array(nll)})

                # plot the transformation (SLOW)
                # plot_joint(data, "blue", f"Transform, Layer: {i}", True)

        return data, X_ldj, forward_funcs, inverse_funcs, it_red

    # run iterations
    data, X_ldj, forward_funcs, inverse_funcs, it_red = iterate(
        data.T, wandb.config.n_layers
    )

    # plot Gaussianized data
    plot_joint(data, "red", title="Gaussianized Data", logger=True)

    # Plot total correlation
    plot_total_corr(
        np.cumsum(np.array(it_red)),
        f"Information Reduction, TC:{np.sum(it_red):.4f}",
        logger=True,
    )

    # TODO: Plot Samples Drawn
    # TODO: Plot Probabilities


if __name__ == "__main__":
    main()
