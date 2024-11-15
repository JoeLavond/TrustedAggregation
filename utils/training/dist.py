# packages
from typing import *
import numpy as np


# divergence
def empirical_cdfs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical cumulative distribution functions for two samples
    Determine the values at which the CDFs are evaluated

    Args:
        x (np.ndarray): first sample
        y (np.ndarray): second sample

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: CDFs for x and y, values at which CDFs are evaluated
    """

    values = np.append(
        np.unique(x),
        np.unique(y)
    )
    values = np.sort(np.unique(values))

    x_cdf = np.array([np.sum(x <= val) / len(x) for val in values])
    y_cdf = np.array([np.sum(y <= val) / len(y) for val in values])

    return x_cdf, y_cdf, values


def ks_div(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Kolmogorov-Smirnov divergence between two samples
    Determine the maximum difference between the two empirical CDFs

    Args:
        x (np.ndarray): first sample
        y (np.ndarray): second sample

    Returns:
        float: Kolmogorov-Smirnov divergence between x and y

    """
    (x_cdf, y_cdf, _) = empirical_cdfs(x, y)
    return np.max(np.abs(y_cdf - x_cdf))
