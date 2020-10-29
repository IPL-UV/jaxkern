from typing import Callable
import jax
import jax.numpy as np
import tqdm
import objax
from objax.typing import JaxArray
from objax.optimizer import SGD


def optimize_model(
    model,
    loss_f: Callable[[JaxArray, JaxArray], JaxArray],
    X,
    y,
    lr: float = 0.01,
    opt=SGD,
    n_epochs: int = 100,
    jitted: bool = True,
) -> None:

    # create optimizer for variables
    opt = opt(model.vars())

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

    for epoch in tqdm.trange(n_epochs):

        # Train
        loss = train_op(X, y.squeeze())

        losses.append(loss)

    return model, losses
