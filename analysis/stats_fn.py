from typing import Tuple
import numpy as np


def compute_snr(data: np.ndarray, axis: int | Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the signal-to-noise ratio of input data over specified axis/axes.

    SNR is defined as the mean divided by the standard deviation over the given axis.
    Also returns the standard error of the SNR estimate assuming **normality** and **independence**.

    :param data: np.ndarray of arbitrary shape. The axis/axes over which to compute the average must be valid.
    :param axis: int or tuple of ints, axis or axes over which to compute mean and std.
    :return: (snr, snr_se) -- arrays with shape reduced over `axis`
    """
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=1)

    if isinstance(axis, int):
        n = data.shape[axis]
    else:
        n = np.prod([data.shape[a] for a in axis])

    snr = mean / (std + 1e-10)
    var_snr = (1.0 / n) + (snr ** 2) / (2.0 * n)
    snr_se = np.sqrt(var_snr)
    return snr, snr_se


def compute_mean_sem(data: np.ndarray, axis: int | Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the mean and standard error of the mean (SEM) over specified axis/axes.

    :param data: np.ndarray of arbitrary shape.
    :param axis: int or tuple of ints, axis/axes over which to compute mean and SEM.
    :return: (mean, sem) arrays with shape reduced over `axis`
    """
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=1)

    if isinstance(axis, int):
        n = data.shape[axis]
    else:
        n = np.prod([data.shape[a] for a in axis])

    sem = std / (np.sqrt(n) + 1e-10)
    return mean, sem


def calc_cohen_d(m1: np.ndarray, m2: np.ndarray, axis: int | Tuple[int, ...] = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute unpaired Cohen's d and approximate 95% confidence interval over specified axis.

    :param m1: First sample array.
    :param m2: Second sample array. Must be broadcast-compatible with m1.
    :param axis: Axis or axes to compute mean/std across.
    :return: Tuple of (cohen_d, 95% CI half-width)
    """
    mean1 = np.mean(m1, axis=axis)
    mean2 = np.mean(m2, axis=axis)
    var1 = np.var(m1, axis=axis, ddof=1)
    var2 = np.var(m2, axis=axis, ddof=1)

    if isinstance(axis, int):
        n1 = m1.shape[axis]
        n2 = m2.shape[axis]
    else:
        n1 = np.prod([m1.shape[a] for a in axis])
        n2 = np.prod([m2.shape[a] for a in axis])

    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2) + 1e-10)
    d = (mean1 - mean2) / pooled_sd

    se_d = np.sqrt((n1 + n2) / (n1 * n2) + (d ** 2) / (2 * (n1 + n2)))
    ci = 1.96 * se_d  # Approximate 95% confidence interval
    return d, ci


def mean_ci(data: np.ndarray, axis: int | Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the mean and confidence interval of the data.

    :param data: np.ndarray of arbitrary shape.
    :param axis: int or tuple of ints, axis/axes over which to compute mean and CI.
    """
    mean = np.mean(data, axis=axis)
    if isinstance(axis, int):
        n = data.shape[axis]
    else:
        n = np.prod([data.shape[a] for a in axis])
    std_err = np.std(data, axis=axis) / np.sqrt(n)
    ci = 1.96 * std_err  # For a 95% confidence interval
    return mean, ci
