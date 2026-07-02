"""Fit calibrated MN sizes back onto the full MN population."""

from __future__ import annotations

import numpy as np

from Regression import regression


def calibrated_sizes_func(Real_MU_pop, Calib_sizes_final, muscle, MN_pop):
    """Fit the original size-distribution law to calibrated sizes."""
    real_pop = np.asarray(Real_MU_pop, dtype=float).ravel()
    sizes = np.asarray(Calib_sizes_final, dtype=float).ravel()
    popt, r2 = regression(real_pop, sizes, "size", muscle, MN_pop)
    return real_pop, sizes, popt, r2
