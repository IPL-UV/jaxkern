import jax
import jax.numpy as np


def histogram_entropy(data, base=2):
    """Calculates the histogram entropy of 1D data.
    This function uses the histogram and then calculates
    the entropy. Does the miller-maddow correction
    
    Parameters
    ----------
    data : np.ndarray, (n_samples,)
        the input data for the entropy
    
    base : int, default=2
        the log base for the calculation.
    
    Returns
    -------
    S : float
        the entropy"""
    # get number of samples
    n_samples = np.shape(data)[0]

    # get number of bins (default square root heuristic)
    nbins = int(np.ceil(np.sqrt(n_samples)))

    # get histogram counts and bin edges
    counts, bin_edges = np.histogram(data, bins=nbins, density=False)

    # get bin centers and sizes
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)

    # get difference between the bins
    delta = bin_centers[3] - bin_centers[2]

    # normalize counts (density)
    pk = 1.0 * np.array(counts) / np.sum(counts)

    # calculate the entropy
    S = entropy(pk, base=base)

    # Miller Maddow Correction
    correction = 0.5 * (np.sum(counts > 0) - 1) / counts.sum()

    return S + correction + np.log2(delta)


def entropy(pk: np.ndarray, base: int = 2) -> np.ndarray:
    """calculate the entropy
    
    Notes
    -----
    Source of this module is the scipy entropy
    module which can be found - shorturl.at/pyABR
    """
    # calculate entropy
    vec = jax.scipy.special.entr(pk)

    # sum all values
    S = np.sum(vec)

    # change base
    S /= np.log(base)

    return S
