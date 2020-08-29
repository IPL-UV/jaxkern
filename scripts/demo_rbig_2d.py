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
from rbig_jax.plots.prob import plot_joint_prob
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
    wandb.config.alpha = 0.0
    wandb.config.n_layers = 20

    # =========================
    # Original Data
    # =========================

    data = get_classic(wandb.config.n_samples)

    # ========================
    # PLOT
    # ========================
    plot_joint(data, "blue", "Original Data", True)

    # ========================
    # Init Transformation
    # ========================

    # initialize parameters getter function
    init_func = init_params_hist(
        wandb.config.support_extension, wandb.config.precision, wandb.config.alpha,
    )

    # define step function (updates the parameters)
    step = partial(get_rbig_params, init_func=init_func)

    def iterate(data, n_layers=10):

        # initialize the log determinant jacobian transforms
        X_ldj = np.zeros(data.shape)

        # prepare to grab the items
        forward_funcs = list()
        inverse_funcs = list()
        it_red = list()

        # loop through the number of layers
        X_transform = data
        with tqdm.trange(n_layers) as pbar:
            for i in pbar:

                # step through
                X_transform_, iX_ldj, forward_func, inverse_func = step(X_transform)

                # calculate the information loss
                it = information_reduction(X_transform, X_transform_)
                wandb.log({"Delta Multi-Information": onp.array(it)})

                # calculate the running total corrlation
                it_red.append(it)
                tc = np.array(it_red).sum()
                wandb.log({"TC": onp.array(tc)})

                # save functions
                forward_funcs.append(forward_func)
                inverse_funcs.append(inverse_func)

                # update data and ldj
                X_transform = X_transform_
                X_ldj += iX_ldj

                # calculate negative log likelihood
                nll = jax.scipy.stats.norm.logpdf(data) + X_ldj
                nll = nll.sum(axis=1).mean()
                wandb.log({"Negative Log-Likelihood": onp.array(nll)})

                # plot the transformation (SLOW)
                # plot_joint(data, "blue", f"Transform, Layer: {i}", True)

        return X_transform, X_ldj, forward_funcs, inverse_funcs, it_red

    # run iterations
    X_transform, X_ldj, forward_funcs, inverse_funcs, it_red = iterate(
        data, wandb.config.n_layers
    )

    # plot Gaussianized data
    plot_joint(X_transform, "red", title="Gaussianized Data", logger=True)

    # ==============================
    # Forward Transformation
    # ==============================
    X_transform = data
    for i in range(wandb.config.n_layers):

        X_transform, _ = forward_funcs[i](X_transform)
    # plot Gaussianized data
    plot_joint(X_transform, "orange", title="Gaussianized Data (Forward)", logger=True)

    # ==============================
    # Inverse Transformation
    # ==============================
    # propagate through inverse function
    inv_funcs = inverse_funcs[::-1]

    X_approx = X_transform
    for i in range(wandb.config.n_layers):

        X_approx = inv_funcs[i](X_approx)

    plot_joint(X_approx, "blue", title="Inverse Transformation", logger=True)

    # ==============================
    # Sampling
    # ==============================
    # draw samples
    gauss_samples = onp.random.randn(data.shape[0], data.shape[1])

    X_approx = np.array(gauss_samples)

    for i in range(wandb.config.n_layers):

        X_approx = inv_funcs[i](X_approx)

    plot_joint(X_approx, "green", title="Samples Drawn", logger=True)

    # ==============================
    # Stopping Criteria
    # ==============================
    # Plot total correlation
    plot_total_corr(
        np.cumsum(np.array(it_red)),
        f"Information Reduction, TC:{np.sum(it_red):.4f}",
        logger=True,
    )
    # ==============================
    # Probability
    # ==============================
    # Plot Log Probability
    log_probs = jax.scipy.stats.norm.logpdf(data) + X_ldj

    log_probs = log_probs.sum(axis=1)

    # clip probabilities
    log_probs_clipped = np.clip(log_probs, -20, 0)
    plot_joint_prob(
        data, log_probs_clipped, "Reds", title="Log Probability", logger=True
    )

    # Plot Probability
    probs = np.exp(log_probs)

    # clip probabilities
    probs_clipped = np.clip(probs, 0, 1)
    plot_joint_prob(data, probs_clipped, "Reds", title="Probability", logger=True)


if __name__ == "__main__":
    main()
