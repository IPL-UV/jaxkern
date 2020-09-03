from typing import Tuple, Union

import jax.numpy as np


def get_domain_extension(
    data: np.ndarray, extension: Union[float, int],
) -> Tuple[float, float]:
    """Gets the extension for the support
    
    Parameters
    ----------
    data : np.ndarray
        the input data to get max and minimum

    extension : Union[float, int]
        the extension
    
    Returns
    -------
    lb : float
        the new extended lower bound for the data
    ub : float
        the new extended upper bound for the data
    """

    # case of int, convert to float
    if isinstance(extension, int):
        extension = float(extension / 100)

    # get the domain
    domain = np.abs(np.max(data) - np.min(data))

    # extend the domain
    domain_ext = extension * domain

    # get the extended domain
    lb = np.min(data) - domain_ext
    up = np.max(data) + domain_ext

    return lb, up
