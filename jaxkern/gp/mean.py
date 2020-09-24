import jax.numpy as jnp


def zero_mean(x):
    """Mean Function"""
    return jnp.zeros(x.shape[0])
