import jax.numpy as np


def get_quantiles(x, n_quantiles: int = 1_000):

    # create outputs (p=[0,1])
    references = np.linspace(0, 1, num=np.maximum(n_quantiles, x.shape[0]))

    # get quantiles
    quantiles = np.quantile(x, references, axis=0)

    return quantiles, references
