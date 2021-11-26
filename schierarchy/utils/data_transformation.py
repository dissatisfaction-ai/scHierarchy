import numpy as np


def data_to_zero_truncated_cdf(x):
    x_sorted = np.sort(x)
    indx_sorted = np.argsort(x)
    x_sorted = x[indx_sorted]
    zero_ind = np.where(x_sorted == 0)[0][-1]
    p = np.concatenate(
        [
            np.zeros(zero_ind),
            1.0 * np.arange(len(x_sorted) - zero_ind) / (len(x_sorted) - zero_ind - 1),
        ]
    )
    cdfs = np.zeros_like(x)
    cdfs[indx_sorted] = p
    return cdfs
