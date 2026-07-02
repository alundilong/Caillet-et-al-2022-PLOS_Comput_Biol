"""
Vectorized cumulative spike-train construction.

This is a drop-in replacement for the original CST_func. It preserves the
function signature and return values, but avoids nested Python loops over every
spike event by using NumPy advanced indexing.
"""

import numpy as np


def _as_sample_indices(spike_times, unit="sample", fs=2048):
    arr = np.asarray(spike_times, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if unit == "sec":
        arr = arr * fs
    elif unit != "sample":
        raise ValueError("unit must be 'sample' or 'sec'")
    return arr.astype(np.int64)


def CST_func(Nb_MN, time, disch_times, unit="sample", fs=2048):
    """
    Compute binary discharge matrix and cumulative spike train (CST).

    Parameters are identical to the original implementation. Discharge times can
    be supplied in sample indices or seconds. Out-of-range events are ignored.

    Returns
    -------
    Binary_matrix : ndarray, shape (Nb_MN, len(time))
        Binary matrix with one row per motoneuron.
    CST : ndarray, shape (len(time),)
        Number of motoneurons firing at each sample.
    """
    n_time = len(time)
    binary_matrix = np.zeros((int(Nb_MN), n_time), dtype=float)

    for i in range(int(Nb_MN)):
        idx = _as_sample_indices(disch_times[i], unit=unit, fs=fs)
        idx = idx[(idx >= 0) & (idx < n_time)]
        if idx.size:
            # Duplicate spike entries should still be interpreted as one spike
            # at that sample, consistent with the original assignment behavior.
            binary_matrix[i, np.unique(idx)] = 1.0

    cst = binary_matrix.sum(axis=0)
    return binary_matrix, cst
