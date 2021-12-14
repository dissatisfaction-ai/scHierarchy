import numpy as np
from scipy import stats


def data_to_zero_truncated_cdf(x):
    """
    Quantile transformation of expression. To convert matrix apply along cell dimension
    Parameters
    ----------
    x: log1p(expression / total_expression * 1e4) transformed expression vector (for 1 gene)

    Returns
    -------
    quantile normalised expression vector
    """
    x_sorted = np.sort(x)
    indx_sorted = np.argsort(x)
    x_sorted = x[indx_sorted]
    try:
        zero_ind = np.where(x_sorted == 0)[0][-1]
        p = np.concatenate(
            [
                np.zeros(zero_ind),
                1.0
                * np.arange(len(x_sorted) - zero_ind)
                / (len(x_sorted) - zero_ind - 1),
            ]
        )
    except IndexError:
        p = 1.0 * np.arange(len(x_sorted)) / (len(x_sorted))
    cdfs = np.zeros_like(x)
    cdfs[indx_sorted] = p
    return cdfs


def cdf_to_pseudonorm(cdf, clip=1e-3):
    """
    Parameters
    ----------
    cdf: Quantile transformed expression
    clip: Clipping normal distriubtion (position of zeros)

    Returns
    -------
    Pseudonormal expression
    """
    return stats.norm.ppf(np.maximum(np.minimum(cdf, 1.0 - clip), clip))


def variance_normalistaion(x, std, clip=10):
    """
    Normalisation by standard deviation
    Parameters
    ----------
    x: Expression data
    std: std in the right dimensions
    clip: clipping threshold for standard deviation

    Returns
    -------

    """
    if clip is not None:
        return x / np.minimum(std, clip)
    else:
        return x / np.minimum(std)
