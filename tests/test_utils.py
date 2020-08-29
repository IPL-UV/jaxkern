import jax.numpy as np
import numpy as onp
import pytest
from jax import random

from src.utils import searchsorted


def test_searchsorted():

    # create 1d array
    bin_locations = np.linspace(0, 1, 10)

    left_boundaries = bin_locations[:-1]
    right_boundaries = bin_locations[:-1] + 0.1
    mid_points = bin_locations[:-1] + 0.05
    print("Bins:", bin_locations.shape)

    for inputs in [left_boundaries, right_boundaries, mid_points]:
        idx = searchsorted(bin_locations[None, :], left_boundaries)

        onp.testing.assert_array_equal(onp.array(idx), onp.arange(0, 9))


def test_searchsorted_arbitrary_shape():
    shape = [2, 3, 4]
    bin_locations = np.linspace(0, 1, 10)
    bin_locations = np.tile(bin_locations, reps=(*shape, 1))

    # initialize random
    key = random.PRNGKey(0)
    inputs = random.uniform(key, shape=(*shape,))
    print(inputs.shape, bin_locations.shape)

    idx = searchsorted(bin_locations, inputs)

    onp.testing.assert_equal(inputs.shape, idx.shape)


def test_searchsorted_double():

    bin_locations = np.linspace(0, 1, 10)
    bin_locations = np.tile(bin_locations, reps=(2, 1))
