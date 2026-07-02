"""First-firing-time error after final calibrated pool simulation."""

from __future__ import annotations

import numpy as np


def delta_ft1_calib_func(Nb_MN, Real_MN_pop, Firing_times_sim, THRESHOLDS):
    """Return simulated first firing time minus experimental first time."""
    n_mn = int(Nb_MN)
    real_pop = np.asarray(Real_MN_pop, dtype=int).ravel()
    first_times = np.zeros(n_mn, dtype=float)
    for i in range(n_mn):
        sim_idx = real_pop[i]
        spikes = np.asarray(Firing_times_sim[sim_idx], dtype=float).ravel()
        first_times[i] = spikes[0] if spikes.size > 2 else 0.0
    return first_times - np.asarray(THRESHOLDS[:n_mn, 0], dtype=float)
