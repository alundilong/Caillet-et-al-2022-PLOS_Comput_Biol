"""Shared utilities for the refactored MN modules.

The functions in this file are deliberately small and dependency-light because
all legacy modules are imported by adding ``MN_Modules`` to ``sys.path`` rather
than as a package.  Keep imports absolute in the public modules for drop-in
compatibility with the original scripts.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def as_1d_float_array(values) -> np.ndarray:
    """Return finite values as a 1-D float array."""
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def as_1d_int_array(values) -> np.ndarray:
    """Return finite values as a 1-D int64 array."""
    return as_1d_float_array(values).astype(np.int64)


def object_array(length: int, fill=None) -> np.ndarray:
    """Create a 1-D object array, optionally filled with one value."""
    out = np.empty(int(length), dtype=object)
    if fill is not None:
        out[:] = fill
    return out


def safe_corrcoef(x, y) -> float:
    """Squared Pearson correlation with robust NaN handling.

    Returns 0.0 when arrays are too short or have zero variance.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    if n < 2:
        return 0.0
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def clip_sample_indices(indices, n_samples: int) -> np.ndarray:
    """Return unique sample indices within [0, n_samples)."""
    idx = as_1d_int_array(indices)
    idx = idx[(idx >= 0) & (idx < int(n_samples))]
    return np.unique(idx)


def get_current_values(I, times: np.ndarray) -> np.ndarray:
    """Evaluate a current-input callable on a vector of times.

    The refactored main pipeline provides an array-backed callable with a
    ``values_at`` method.  For legacy callables, fall back to a Python loop.
    """
    if hasattr(I, "values_at"):
        return np.asarray(I.values_at(times), dtype=float)
    return np.fromiter((float(I(float(t))) for t in times), dtype=float, count=len(times))
