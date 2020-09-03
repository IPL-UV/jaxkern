import jax
import jax.numpy as np


@jax.jit
def interp_dim(x_new, x, y):
    return jax.vmap(np.interp, in_axes=(0, 0, 0))(x_new, x, y)


def searchsorted(bin_locations, inputs, eps=1e-6):
    # add noise to prevent zeros
    # bin_locations = bin_locations[..., -1] + eps
    bin_locations = bin_locations + eps

    # find bin locations (parallel bisection search)

    # sum dim
    print("Bins:", bin_locations.shape)
    print("Inputs:", inputs[..., None].shape)
    input_bins = np.sum(inputs[..., None] >= bin_locations, axis=-1)

    return input_bins
