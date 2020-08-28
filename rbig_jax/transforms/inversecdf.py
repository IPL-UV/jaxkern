import jax
import jax.numpy as np


def forward_inversecdf_1d(X):
    return jax.scipy.stats.norm.ppf(X)


def inverse_inversecdf_1d(X):
    return jax.scipy.stats.norm.cdf(X)
