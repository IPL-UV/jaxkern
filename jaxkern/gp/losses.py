from functools import partial
from typing import Callable, Dict, Tuple

import objax
from objax.typing import JaxArray
import jax
import jax.numpy as np


def negative_log_likelihood(
    gp_model: objax.Module, X: JaxArray, y: JaxArray
) -> JaxArray:
    """Negative Log-Likelihood given a GP model

    This function is meant to act as a helper to the
    exact GP model. So given a GP model fitted to some inputs,
    we get a distribution where we can calculate the log-likelihood
    of the distribution conditioned on the outputs y.

    Parameters
    ----------
    gp_model: objax.Module
        the GP model which outputs a distribution.

    X : JaxArray
        the inputs (n_samples, n_features)

    Y : JaxArray
        the inputs, (n_samples, 1)

    Returns
    -------
    nll : JaxArray
        the negative log-likelihood value, ()
    """
    # construct a GP model
    dist = gp_model.forward(X)

    # return the negative log-likelihood
    return -dist.log_prob(y.T).mean()
