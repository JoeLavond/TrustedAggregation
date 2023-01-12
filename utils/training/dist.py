# packages
import numpy as np


# divergence
def empirical_cdfs(x, y):

    values = np.append(
        np.unique(x),
        np.unique(y)
    )
    values = np.sort(np.unique(values))

    x_cdf = np.array([np.sum(x <= val) / len(x) for val in values])
    y_cdf = np.array([np.sum(y <= val) / len(y) for val in values])

    return x_cdf, y_cdf, values


def ks_div(x, y):
    (x_cdf, y_cdf, _) = empirical_cdfs(x, y)
    return np.max(np.abs(y_cdf - x_cdf))
