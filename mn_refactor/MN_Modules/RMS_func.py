"""Root-mean-square error helper."""

from __future__ import annotations

import numpy as np


def RMS_func(arr_exp, arr_simul):
    """Compute RMS(arr_exp - arr_simul) after length alignment."""
    x = np.asarray(arr_exp, dtype=float).ravel()
    y = np.asarray(arr_simul, dtype=float).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[:n] - y[:n]) ** 2)))
