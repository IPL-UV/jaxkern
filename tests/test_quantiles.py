import pytest
import jax.numpy as np
from src.quantiles import get_quantiles


# def test_quantiles_1d():

#     # create 1d array
#     bin_locations = np.linspace(0, 1, 10)

#     # boundaries
#     left_boundaries = bin_locations[:-1]

#     # get quantiles
#     quantiles, references = get_quantiles(bin_locations, 10)
