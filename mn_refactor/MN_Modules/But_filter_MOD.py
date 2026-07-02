"""Cached 4th-order Butterworth low-pass filtering."""

from functools import lru_cache
import numpy as np
from scipy import signal


@lru_cache(maxsize=32)
def _butter_coefficients(fc, fs):
    w = float(fc) / (float(fs) / 2.0)
    return signal.butter(4, w, "low", analog=False)


def But_filter_func(fc, raw_signal, fs=2048):
    """Apply the original 4th-order zero-phase Butterworth low-pass filter."""
    b, a = _butter_coefficients(float(fc), float(fs))
    return signal.filtfilt(b, a, np.asarray(raw_signal, dtype=float))
