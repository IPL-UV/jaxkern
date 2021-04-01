import sys, os
from pyprojroot import here
from wandb.sdk import wandb_config

# spyder up to find the root
root = here(project_files=[".here"])
cwd = os.getcwd()
# append to path
sys.path.append(str(root))

from jaxkern.viz import plot_1D_GP
from jaxkern.gp.uncertain.mcmc import MCMomentTransform, init_mc_moment_transform
from jaxkern.gp.uncertain.unscented import UnscentedTransform, init_unscented_transform
from jaxkern.gp.uncertain.linear import init_taylor_transform, init_taylor_o2_transform
from jaxkern.gp.uncertain.mcmc import MCMomentTransform
from jaxkern.gp.uncertain.unscented import UnscentedTransform, SphericalRadialTransform
from jaxkern.gp.uncertain.quadrature import GaussHermite
from jaxkern.gp.uncertain.predict import moment_matching_predict_f


# jax packages
import itertools

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.config import config
from jax import device_put
from jax import random
import numpy as np

# import chex
config.update("jax_enable_x64", True)


# logging
import tqdm
import wandb

# plot methods
import matplotlib.pyplot as plt
import seaborn as sns

sns.reset_defaults()
sns.set_context(context="talk", font_scale=0.7)
from argparse import ArgumentParser
import wandb


# ==========================
# PARAMETERS
# ==========================

parser = ArgumentParser(
    description="2D Data Demo with Iterative Gaussianization method"
)

# ======================
# Dataset
# ======================
parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="number of data points for training",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="classic",
    help="Dataset to be used for visualization",
)
parser.add_argument(
    "--n_train",
    type=int,
    default=60,
    help="number of data points for training",
)
parser.add_argument(
    "--n_test",
    type=int,
    default=100,
    help="number of data points for testing",
)
parser.add_argument(
    "--y_noise",
    type=float,
    default=0.05,
    help="number of data points for training",
)
parser.add_argument(
    "--x_noise",
    type=float,
    default=0.3,
    help="number of data points for training",
)
# ======================
# Model Training
# ======================
parser.add_argument("--epochs", type=int, default=2_000, help="Number of batches")
parser.add_argument(
    "--learning_rate", type=float, default=0.005, help="Number of batches"
)
# ======================
# MC Parameters
# ======================
parser.add_argument("--mc_samples", type=int, default=1_000, help="Number of batches")
# ======================
# Sigma Parameters
# ======================
parser.add_argument("--alpha", type=float, default=1.0, help="Number of batches")
parser.add_argument("--beta", type=float, default=2.0, help="Number of batches")
parser.add_argument("--kappa", type=float, default=None, help="Number of batches")
# ======================
# Quadrature
# ======================
parser.add_argument("--degree", type=int, default=20, help="Number of batches")

# ======================
# Logger Parameters
# ======================
parser.add_argument("--name", type=str, default="gp_mm")
parser.add_argument("--wandb-entity", type=str, default="ipl_uv")
parser.add_argument("--wandb-project", type=str, default="jaxkern-demos")
# =====================
# Testing
# =====================
parser.add_argument(
    "-sm",
    "--smoke-test",
    action="store_true",
    help="to do a smoke test without logging",
)

args = parser.parse_args()
# change this so we don't bug wandb with our BS
if args.smoke_test:
    os.environ["WANDB_MODE"] = "dryrun"
    args.epochs = 1
    args.n_samples = 1_000

# ==========================
# INITIALIZE LOGGER
# ==========================

wandb_logger = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
wandb_logger.config.update(args)
config = wandb_logger.config


# ===================================
# TRAINING DATA
# ===================================
key = jax.random.PRNGKey(config.seed)
y_noise = config.y_noise
x_noise = config.x_noise

f = lambda x: jnp.sin(1.0 * jnp.pi / 1.6 * jnp.cos(5 + 0.5 * x))
ntrain = config.n_train

X = jnp.linspace(-10, 10, ntrain).reshape(-1, 1)

key, y_rng = jax.random.split(key, 2)
y = f(X)

# Noisy Signal

key, x_rng = jax.random.split(key, 2)

X_noise = X + x_noise * jax.random.normal(x_rng, shape=X.shape)
y_noise = f(X) + y_noise * jax.random.normal(y_rng, shape=X.shape)

# sort inputs
idx_sorted = jnp.argsort(X_noise, axis=0).squeeze()

X_noise = X_noise[(idx_sorted,)]
y_noise = y_noise[(idx_sorted,)]
X = X[(idx_sorted,)]
y = y[(idx_sorted,)]

ntest = config.n_test


Xtest = jnp.linspace(-10.1, 10.1, ntest)[:, None]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))
ax[0].scatter(X, y, color="red")
ax[1].scatter(X, y_noise, color="red")
ax[2].scatter(X_noise, y_noise, color="red")
wandb.log({"data_train": wandb.Image(plt)})
plt.show()

# ==========================
# GP MODEL
# ==========================
from gpjax.gps import Prior
from gpjax.mean_functions import Zero
from gpjax.kernels import RBF
from gpjax.likelihoods import Gaussian
from gpjax.types import Dataset


# GP Prior
mean_function = Zero()
kernel = RBF()
prior = Prior(mean_function=mean_function, kernel=kernel)

# GP Likelihood
lik = Gaussian()

# GP Posterior
posterior = prior * lik

# initialize training dataset
training_ds = Dataset(X=X, y=y_noise)

# PARAMETERS
from gpjax.parameters import initialise
import numpyro.distributions as dist
from gpjax.interfaces.numpyro import numpyro_dict_params, add_constraints


# initialize parameters
params = initialise(posterior)


hyperpriors = {
    "lengthscale": 1.0,
    "variance": 1.0,
    "obs_noise": 0.01,
}


# convert to numpyro-style params
numpyro_params = numpyro_dict_params(hyperpriors)

# convert to numpyro-style params
numpyro_params = add_constraints(numpyro_params, dist.constraints.softplus_positive)

# INFERENCE
from numpyro.infer.autoguide import AutoDelta
from gpjax.interfaces.numpyro import numpyro_marginal_ll, numpyro_dict_params


# initialize numpyro-style GP model
npy_model = numpyro_marginal_ll(posterior, numpyro_params)

# approximate posterior
guide = AutoDelta(npy_model)

# TRAINING
import numpyro
from numpyro.infer import SVI, Trace_ELBO

# reproducibility
key, opt_key = jr.split(key, 2)
n_iterations = config.epochs
lr = config.learning_rate

# numpyro specific optimizer
optimizer = numpyro.optim.Adam(step_size=lr)

# stochastic variational inference (pseudo)
svi = SVI(npy_model, guide, optimizer, loss=Trace_ELBO())
svi_results = svi.run(opt_key, n_iterations, training_ds)

# Learned Params
learned_params = svi_results.params
p = jax.tree_map(lambda x: np.array(x), learned_params)
wandb.log(p)

# ==============================
# PREDICTIONS (CLEAN)
# ==============================
from gpjax import mean, variance

meanf = mean(posterior, learned_params, training_ds)
covarf = variance(posterior, learned_params, training_ds)
varf = lambda x: jnp.atleast_1d(jnp.diag(covarf(x)))


mu = meanf(Xtest).squeeze()
var = varf(Xtest).squeeze()


plot_1D_gp_clean = jax.partial(plot_1D_GP, X=X, y=y, Xtest=Xtest)

fig, ax = plot_1D_gp_clean(ytest=mu, y_mu=mu, y_var=var)
wandb.log({"preds_standard_clean": wandb.Image(plt)})

# ==============================
# PREDICTIONS (NOISY)
# ==============================

input_cov = jnp.array([x_noise]).reshape(-1, 1)  # ** 2


Xtest = jnp.linspace(-10.1, 10.1, ntest)[:, None]
ytest = f(Xtest)

demo_sample_idx = 47

key, xt_rng = jax.random.split(key, 2)

Xtest_noisy = Xtest + x_noise * jax.random.normal(xt_rng, shape=Xtest.shape)


idx_sorted = jnp.argsort(Xtest_noisy, axis=0)

# Xtest = Xtest[(idx_sorted,)]
Xtest_noisy = Xtest_noisy[(idx_sorted,)][..., 0]
ytest_noisy = ytest[(idx_sorted,)]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
ax[1].scatter(Xtest_noisy, ytest_noisy, color="red")
ax[0].scatter(Xtest, ytest, color="red")
ax[1].scatter(
    Xtest_noisy[demo_sample_idx],
    ytest_noisy[demo_sample_idx],
    marker=".",
    s=300,
    color="black",
)
wandb.log({"data_test_noisy": wandb.Image(plt)})

plt.show()


plot_1D_gp_noisy = jax.partial(
    plot_1D_GP, X=Xtest_noisy, y=ytest_noisy, Xtest=Xtest_noisy
)


mu = meanf(Xtest_noisy).squeeze()
var = varf(Xtest_noisy).squeeze()


# fig, ax = plot_1D_GP_noisy(Xtest_noisy, mu, var)
fig, ax = plot_1D_gp_noisy(ytest=mu, y_mu=mu, y_var=var)

wandb.log({"preds_standard_noisy": wandb.Image(plt)})


# ==============================
# MM - MC - PREDICTIONS (NOISY)
# ==============================

mm_transform = MCMomentTransform(n_features=1, n_samples=1_000, seed=123)
# mm_transform = UnscentedTransform(n_features=1, alpha=1.0, beta=2.0, kappa=None)
# mm_transform = GaussHermite(n_features=1, degree=20)
# mm_transform = SphericalRadialTransform(n_features=1)

# init function
n_features = 1
mc_samples = config.mc_samples
covariance = False

mm_transform = MCMomentTransform(
    n_features=n_features, n_samples=mc_samples, seed=config.seed
)

mm_predict_f = moment_matching_predict_f(
    posterior, learned_params, training_ds, mm_transform, obs_noise=True
)

mm_mean_f = jax.vmap(mm_predict_f, in_axes=(0, None))

mu, var = mm_mean_f(Xtest_noisy, input_cov)

fig, ax = plot_1D_gp_noisy(ytest=mu, y_mu=mu, y_var=var.squeeze())
wandb.log({"preds_mc_noisy": wandb.Image(plt)})


# ==============================
# Unscented - PREDICTIONS (NOISY)
# ==============================

# init function
alpha = config.alpha
beta = config.beta
kappa = config.kappa

mm_transform = UnscentedTransform(n_features=1, alpha=alpha, beta=beta, kappa=kappa)
# mm_transform = GaussHermite(n_features=1, degree=20)
# mm_transform = SphericalRadialTransform(n_features=1)


mm_predict_f = moment_matching_predict_f(
    posterior, learned_params, training_ds, mm_transform, obs_noise=True
)

mm_mean_f = jax.vmap(mm_predict_f, in_axes=(0, None))

mu, var = mm_mean_f(Xtest_noisy, input_cov)

fig, ax = plot_1D_gp_noisy(ytest=mu, y_mu=mu, y_var=var.squeeze())
wandb.log({"preds_unscented_noisy": wandb.Image(plt)})


# ==============================
# GaussHermite - PREDICTIONS (NOISY)
# ==============================

# init function
degree = config.degree

mm_transform = GaussHermite(n_features=1, degree=degree)
# mm_transform = SphericalRadialTransform(n_features=1)


mm_predict_f = moment_matching_predict_f(
    posterior, learned_params, training_ds, mm_transform, obs_noise=True
)

mm_mean_f = jax.vmap(mm_predict_f, in_axes=(0, None))

mu, var = mm_mean_f(Xtest_noisy, input_cov)

fig, ax = plot_1D_gp_noisy(ytest=mu, y_mu=mu, y_var=var.squeeze())
wandb.log({"preds_gh_noisy": wandb.Image(plt)})

# ==============================
# Unscented Spherical - PREDICTIONS (NOISY)
# ==============================

mm_transform = SphericalRadialTransform(n_features=1)


mm_predict_f = moment_matching_predict_f(
    posterior, learned_params, training_ds, mm_transform, obs_noise=True
)

mm_mean_f = jax.vmap(mm_predict_f, in_axes=(0, None))

mu, var = mm_mean_f(Xtest_noisy, input_cov)

fig, ax = plot_1D_gp_noisy(ytest=mu, y_mu=mu, y_var=var.squeeze())
wandb.log({"preds_sphere_noisy": wandb.Image(plt)})