import jax
import jax.numpy as np
from functools import partial
from rbig_jax.transforms.gaussian import init_params_hist, get_gauss_params
from rbig_jax.transforms.linear import init_pca_params


def get_rbig_params(data, init_func):

    # get gaussian params
    (
        X_transform,
        X_ldj_mg,
        mg_params,
        forward_mg_func,
        inverse_mg_func,
    ) = get_gauss_params(data.T, init_func)

    # get rotation parameters
    X_transform, rot_params, forward_rot_func, inverse_rot_func = init_pca_params(
        X_transform.T
    )

    # forward transform
    return (
        X_transform,
        X_ldj_mg.T,
        # jax.jit(
        partial(
            forward_transform,
            apply_mg=forward_mg_func,
            params=mg_params,
            apply_rot=forward_rot_func,
            # )
        ),
        # jax.jit(
        partial(
            inverse_transform,
            apply_inv_mg=inverse_mg_func,
            params=mg_params,
            apply_inv_rot=inverse_rot_func,
            # )
        ),
    )


# @partial(jax.jit, static_argnums=(1, 2, 3))
def forward_transform(X, apply_mg, params, apply_rot):
    X, X_ldj = apply_mg(X.T, params)
    X, _ = apply_rot(X.T)
    return X, X_ldj


# @partial(jax.jit, static_argnums=(1, 2, 3))
def inverse_transform(X, apply_inv_mg, params, apply_inv_rot):
    X = apply_inv_rot(X)
    X = apply_inv_mg(X.T, params)
    return X.T

