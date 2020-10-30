import jax.numpy as np
import objax


class Kernel(objax.Module):
    def __init__(self, input_dim: int = 1):

        self.input_dim = objax.TrainRef(np.array([input_dim]))
