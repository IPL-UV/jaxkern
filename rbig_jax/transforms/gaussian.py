from functools import partial
import jax
import jax.numpy as np
from rbig_jax.transforms.uniform import (
    forward_uniformization_1d,
    inverse_uniformization_1d,
)
from rbig_jax.transforms.inversecdf import forward_inversecdf_1d, inverse_inversecdf_1d
from rbig_jax.transforms.histogram import get_params
from rbig_jax.transforms.marginal import (
    forward_gaussianization as forward_1d,
    inverse_gaussianization as inverse_1d,
)


def init_params_hist_1d(support_extension=10, precision=100, alpha=1e-5):

    param_getter = jax.jit(
        partial(
            get_params,
            support_extension=support_extension,
            precision=precision,
            alpha=alpha,
        )
    )

    return param_getter


def init_params_hist(support_extension=10, precision=100, alpha=1e-5):

    param_getter = jax.jit(
        jax.vmap(
            partial(
                get_params,
                support_extension=support_extension,
                precision=precision,
                alpha=alpha,
            )
        )
    )
    return param_getter


def get_gauss_params(X, apply_func):
    X, ldX, params = apply_func(X)

    # clip boundaries
    X = np.clip(X, 1e-5, 1.0 - 1e-5)

    X = forward_inversecdf_1d(X)

    log_prob = ldX - jax.scipy.stats.norm.logpdf(X)

    return X, log_prob, params


# @jax.jit
# def forward_gaussianization(X, params):

#     # transform to uniform domain
#     X, Xdj = forward_uniformization(X, params)

#     # clip boundaries
#     X = np.clip(X, 1e-5, 1.0 - 1e-5)

#     # transform to the gaussian domain
#     X = forward_inversecdf(X)

#     log_prob = Xdj - jax.scipy.stats.norm.logpdf(X)

#     return X, log_prob
