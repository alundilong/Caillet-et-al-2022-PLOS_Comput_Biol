"""Pad ragged firing-time arrays with NaN values."""

from __future__ import annotations

import numpy as np


def build_nan_matrix_func(Firing_times_sim, author, trial="exp"):
    """Return a rectangular NaN-padded matrix from ragged firing times."""
    lengths = [len(row) for row in Firing_times_sim]
    max_length = max(lengths) if lengths else 0
    out = np.full((len(Firing_times_sim), max_length), np.nan, dtype=float)
    for i, row in enumerate(Firing_times_sim):
        values = np.asarray(row, dtype=float).ravel()
        out[i, : values.size] = values
    return out
