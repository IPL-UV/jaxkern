import collections
from typing import Union

from functools import partial
import jax
import jax.numpy as np

from rbig_jax.transforms.utils import get_domain_extension

Params = collections.namedtuple(
    "Params", ["support", "quantiles", "support_pdf", "empirical_pdf"]
)


# @partial(jax.jit, static_argnums=(1, 2))
def get_params(
    X: np.ndarray,
    support_extension: Union[int, float] = 10,
    precision: int = 1_000,
    alpha: float = 1e-5,
):
    """Get parameters via the histogram transform
    
    Parameters
    ----------
    X : np.ndarray, (n_samples)
        input to get histogram transformation
    
    support_extension: Union[int, float], default=10
        extend the support by x on both sides
    
    precision: int, default=1_000
        the number of points to use for the interpolation
    
    alpha: float, default=1e-5
        the regularization for the histogram. ensures that
        there are no zeros in the empirical pdf.
    
    Returns
    -------
    X_trans : np.ndarray, (n_samples,)
        the data transformed via the empirical function
    log_dX : np.ndarray, (n_samples,)
        the log pdf of the data
    Params: namedTuple
        a named tuple with the elements needed for the
        forward and inverse transformation
    
    Examples
    --------
    >>> # single set of parameters
    >>> X_transform, params = get_params(x_samples, 10, 1000)
    
    >>> # example with multiple dimensions
    >>> multi_dims = jax.vmap(get_params, in_axes=(0, None, None))
    >>> X_transform, params = multi_dims(X, 10, 1000)
    """
    # get number of samples
    n_samples = np.shape(X)[0]

    # get number of bins (default square root heuristic)
    nbins = int(np.ceil(np.sqrt(n_samples)))

    # get histogram counts and bin edges
    counts, bin_edges = np.histogram(X, bins=nbins)

    # add regularization
    counts = np.array(counts) + alpha

    # get bin centers and sizes
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)
    bin_size = bin_edges[2] - bin_edges[1]

    # =================================
    # PDF Estimation
    # =================================
    # pdf support
    pdf_support = np.hstack(
        (bin_centers[0] - bin_size, bin_centers, bin_centers[-1] + bin_size)
    )
    # empirical PDF
    empirical_pdf = np.hstack((0.0, counts / (np.sum(counts) * bin_size), 0.0))

    # =================================
    # CDF Estimation
    # =================================
    c_sum = np.cumsum(counts)
    cdf = (1 - 1 / n_samples) * c_sum / n_samples

    incr_bin = bin_size / 2

    # ===============================
    # Extend CDF Support
    # ===============================
    lb, ub = get_domain_extension(X, support_extension)

    # get new bin edges
    new_bin_edges = np.hstack((lb, np.min(X), bin_centers + incr_bin, ub,))

    extended_cdf = np.hstack((0.0, 1.0 / n_samples, cdf, 1.0))

    new_support = np.linspace(new_bin_edges[0], new_bin_edges[-1], int(precision))

    uniform_cdf = jax.lax.cummax(
        np.interp(new_support, new_bin_edges, extended_cdf), axis=0
    )

    # Normalize CDF estimation
    uniform_cdf /= np.max(uniform_cdf)

    return (
        np.interp(X, new_support, uniform_cdf),
        np.log(np.interp(X, pdf_support, empirical_pdf)),
        Params(new_support, uniform_cdf, pdf_support, empirical_pdf),
    )

