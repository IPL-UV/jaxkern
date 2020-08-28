import jax
import jax.numpy as np
from rbig_jax.utils import interp_dim

# multidim_interp = jax.jit()


@jax.jit
def forward_uniformization(X, params):
    return (
        interp_dim(X, params.support, params.quantiles),
        np.log(interp_dim(X, params.support_pdf, params.empirical_pdf)),
    )


@jax.jit
def inverse_uniformization(X, params):
    return interp_dim(X, params.quantiles, params.support)


@jax.jit
def forward_uniformization_1d(X, params):
    return (
        np.interp(X, params.support, params.quantiles),
        np.log(np.interp(X, params.support_pdf, params.empirical_pdf)),
    )


@jax.jit
def inverse_uniformization_1d(X, params):
    return np.interp(X, params.quantiles, params.support)
