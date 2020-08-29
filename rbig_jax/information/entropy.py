import jax
import jax.numpy as np


def entropy_marginal(data, base=2):

    # get number of samples
    n_samples = np.shape(data)[0]

    # get number of bins (default square root heuristic)
    nbins = int(np.ceil(np.sqrt(n_samples)))

    # get histogram counts and bin edges
    counts, bin_edges = np.histogram(data, bins=nbins)

    # get bin centers and sizes
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)

    # get difference between the bins
    delta = bin_centers[3] - bin_centers[2]

    # normalize counts
    pk = 1.0 * np.array(counts) / np.sum(counts)

    # calculate the entropy
    vec = jax.scipy.special.entr(pk)

    S = np.sum(vec)

    # change the base
    S /= np.log(base)

    # Miller Maddow Correction
    correction = 0.5 * (np.sum(counts > 0) - 1) / counts.sum()

    return S + correction + np.log2(delta)

