import jax
import jax.numpy as np


@jax.jit
def interp_dim(x_new, x, y):
    return jax.vmap(np.interp, in_axes=(0, 0, 0))(x_new, x, y)


# multidim_interp = jax.jit()


@jax.jit
def forward_transform(X, params):
    return (
        interp_dim(X, params.support, params.quantiles),
        interp_dim(X, params.support_pdf, params.empirical_pdf),
    )


@jax.jit
def inverse_transform(X, params):
    return interp_dim(X, params.quantiles, params.support)


@jax.jit
def forward_transform_1d(X, params):
    return (
        np.interp(X, params.support, params.quantiles),
        np.interp(X, params.support_pdf, params.empirical_pdf),
    )


@jax.jit
def inverse_transform_1d(X, params):
    return np.interp(X, params.quantiles, params.support)
