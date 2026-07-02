"""Cumulative spike train construction."""

from __future__ import annotations

import numpy as np

from _utils import clip_sample_indices


def _spikes_to_samples(spikes, unit: str, fs: float) -> np.ndarray:
    arr = np.asarray(spikes, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if unit == "sec":
        arr = arr * float(fs)
    elif unit != "sample":
        raise ValueError("unit must be 'sample' or 'sec'")
    return arr.astype(np.int64)


def CST_func(Nb_MN, time, disch_times, unit="sample", fs=2048):
    """Build a binary discharge matrix and its cumulative spike train.

    Parameters
    ----------
    Nb_MN : int
        Number of motoneurons / motor units.
    time : array-like
        Time vector defining the output length.
    disch_times : object array/list
        One spike-time vector per MN.  Values are sample indices by default;
        pass ``unit='sec'`` when values are in seconds.

    Returns
    -------
    Binary_matrix : ndarray, shape (Nb_MN, len(time))
    CST : ndarray, shape (len(time),)
    """
    n_mn = int(Nb_MN)
    n_time = len(time)
    binary_matrix = np.zeros((n_mn, n_time), dtype=float)

    for i in range(n_mn):
        idx = _spikes_to_samples(disch_times[i], unit=unit, fs=fs)
        idx = clip_sample_indices(idx, n_time)
        if idx.size:
            binary_matrix[i, idx] = 1.0

    return binary_matrix, binary_matrix.sum(axis=0)
