import jax
import jax.numpy as np
from rbig_jax.information.entropy import entropy_marginal


def information_reduction(X, Y, tol_dimensions=0, p=0.25):
    n_samples, n_dimensions = X.shape
    if tol_dimensions is None or 0:
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_dimensions = np.interp(n_samples, xxx, yyy)
    # calculate the marginal entropy
    hx = jax.vmap(entropy_marginal)(X.T)
    hy = jax.vmap(entropy_marginal)(Y.T)

    # Information content
    I = np.sum(hy) - np.sum(hx)
    II = np.sqrt(np.sum((hy - hx) ** 2))

    if II < np.sqrt(n_dimensions * p * tol_dimensions ** 2) or I < 0:
        I = 0

    return I
