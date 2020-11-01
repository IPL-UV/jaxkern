from typing import Callable, Tuple, List

import jax
import jax.numpy as np
import objax
import tqdm
from objax.optimizer import Adam
from objax.typing import JaxArray

# TODO: make a more explicit model class input


def optimize_model(
    model: objax.Module,
    loss_f: Callable[[objax.Module, JaxArray, JaxArray], JaxArray],
    X: JaxArray,
    y: JaxArray,
    opt: objax.optimizer = Adam,
    lr: float = 0.01,
    n_epochs: int = 100,
    jitted: bool = True,
    **kwargs
) -> Tuple[objax.Module, List[JaxArray]]:
    """Helper training loop for gp model

    Parameters
    ----------
    model : objax.Module
        a GP model
    loss_f : Callable
        a loss function for the inputs and outputs
    X : JaxArray
        the input data (n_samples, n_features)
    y : JaxArray
        the output data (n_samples, n_features)
    opt : objax.optimizer
        any objax optimizer
    lr : float
        the learning rate for the optimizer
    n_epochs: int
        the number of epochs for the training
    jitted: bool, default=True
        whether or not to jit the training process or not
        (it's generally worth doing as it is much faster but
        it's harder to debug if there are errors)
    **kwargs:
        any keyword arguments for the optimizer

    Returns
    -------
    model : objax.Module
        a trained gp model
    losses : List[JaxArray]
        a list of all of the loss values for every step
    """

    # loss_f = jax.partial(loss_f, model)

    # create optimizer for variables
    opt = opt(model.vars(), **kwargs)

    # get gradient
    gv = objax.GradValues(loss_f, model.vars())

    # get training loop
    def train_op(X, y):

        g, v = gv(X, y)  # returns gradients, loss
        opt(lr, g)
        return v

    if jitted:
        train_op = objax.Jit(train_op, gv.vars() + opt.vars())

    losses = []

    pbar = tqdm.trange(n_epochs)

    for _ in pbar:

        # Train
        loss = train_op(X, y.squeeze())

        losses.append(loss)

    return model, losses
