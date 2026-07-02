"""Hanning-window smoothing for instantaneous discharge-frequency impulses."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal


@lru_cache(maxsize=8)
def _hanning_window(fs: float, window_s: float = 0.4):
    length = max(1, int(round(float(window_s) * float(fs))))
    win = signal.windows.hann(length)
    denom = float(np.sum(win))
    if denom == 0:
        denom = 1.0
    return win, denom


def _filter_one(values, fs: float):
    win, denom = _hanning_window(float(fs))
    return signal.convolve(np.asarray(values, dtype=float), win, mode="same") / denom * float(fs)


def Hanning_filter_func(Nb_MU, FF_FULL, fs=2048):
    """Smooth binary IDF impulses with a 400 ms Hanning window.

    Return type is compatible with the original code: a numeric vector for one
    MU, and an object array of vectors for multiple MUs.
    """
    n = int(Nb_MU)
    if n == 1:
        return _filter_one(FF_FULL, fs)

    try:
        stacked = np.vstack([np.asarray(FF_FULL[i], dtype=float) for i in range(n)])
        win, denom = _hanning_window(float(fs))
        filt = signal.convolve(stacked, win[None, :], mode="same") / denom * float(fs)
        out = np.empty((n,), dtype=object)
        for i in range(n):
            out[i] = filt[i]
        return out
    except ValueError:
        out = np.empty((n,), dtype=object)
        for i in range(n):
            out[i] = _filter_one(FF_FULL[i], fs)
        return out
