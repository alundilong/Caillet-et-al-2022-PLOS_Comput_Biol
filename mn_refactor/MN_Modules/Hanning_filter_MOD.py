"""Smooth instantaneous-frequency impulses with a 400 ms Hanning window."""

import numpy as np
from scipy import signal


def Hanning_filter_func(Nb_MU, FF_FULL, fs=2048):
    """
    Drop-in replacement for the original Hanning filter.

    For multiple MUs, the convolution is vectorized when FF_FULL can be stacked
    into a regular 2D array. The return type remains compatible: an object array
    for Nb_MU > 1 and a numeric vector for Nb_MU == 1.
    """
    window_length = 0.4  # s, according to De Luca-style FIDF smoothing
    L = int(window_length * fs)
    hanning_window = signal.windows.hann(L)
    sum_han = np.sum(hanning_window)

    if int(Nb_MU) > 1:
        try:
            data = np.vstack([np.asarray(FF_FULL[i], dtype=float) for i in range(int(Nb_MU))])
            filtered = signal.convolve(data, hanning_window[None, :], mode="same") / sum_han
            filtered = filtered * fs
            out = np.empty((int(Nb_MU),), dtype=object)
            for i in range(int(Nb_MU)):
                out[i] = filtered[i]
            return out
        except ValueError:
            out = np.empty((int(Nb_MU),), dtype=object)
            for i in range(int(Nb_MU)):
                filtered = signal.convolve(np.asarray(FF_FULL[i], dtype=float), hanning_window, mode="same") / sum_han
                out[i] = filtered * fs
            return out

    filtered_single = signal.convolve(np.asarray(FF_FULL, dtype=float), hanning_window, mode="same") / sum_han
    return filtered_single * fs
