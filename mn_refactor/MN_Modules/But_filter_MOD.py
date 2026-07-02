"""Butterworth low-pass filtering used for CST-derived common input/control."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import signal


@lru_cache(maxsize=32)
def _butter_lowpass(fc: float, fs: float, order: int = 4):
    nyquist = fs / 2.0
    if fc <= 0 or fc >= nyquist:
        raise ValueError(f"Cutoff frequency fc={fc} must be in (0, fs/2={nyquist}).")
    return signal.butter(order, fc / nyquist, btype="low", analog=False)


def But_filter_func(fc, raw_signal, fs=2048):
    """Apply the original 4th-order zero-phase low-pass filter.

    Parameters are kept compatible with the original module.  Filter
    coefficients are cached, which avoids repeatedly redesigning the same
    Butterworth filters during a run.
    """
    b, a = _butter_lowpass(float(fc), float(fs), order=4)
    x = np.asarray(raw_signal, dtype=float)
    return signal.filtfilt(b, a, x)
