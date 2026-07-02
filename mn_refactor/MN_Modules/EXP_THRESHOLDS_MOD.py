"""Recruitment and derecruitment threshold extraction."""

from __future__ import annotations

import numpy as np


def exp_thresholds_func(Nb_MN, MVC, disch_times, common_input, Force, fs=2048):
    """Compute timing, common-input, and force thresholds for each MN.

    Returns a matrix with columns:
    first time [s], recruitment %CI, recruitment %MVC,
    last time [s], derecruitment %CI, derecruitment %MVC.
    """
    n_mn = int(Nb_MN)
    ci = np.asarray(common_input, dtype=float).ravel()
    force = np.asarray(Force, dtype=float).ravel()
    max_ci = np.max(ci) if np.max(ci) != 0 else 1.0
    max_force = np.max(force) if np.max(force) != 0 else 1.0
    thresholds = np.zeros((n_mn, 6), dtype=float)

    for i in range(n_mn):
        spikes = np.asarray(disch_times[i], dtype=int).ravel()
        if spikes.size == 0:
            continue
        first = int(np.clip(spikes[0], 0, min(ci.size, force.size) - 1))
        last = int(np.clip(spikes[-1], 0, min(ci.size, force.size) - 1))
        thresholds[i, 0] = round(first / fs, 3)
        thresholds[i, 1] = ci[first] / max_ci * 100.0
        thresholds[i, 2] = force[first] / max_force * float(MVC) * 100.0
        thresholds[i, 3] = round(last / fs, 3)
        thresholds[i, 4] = int(ci[last] / max_ci * 100.0)
        thresholds[i, 5] = int(force[last] / max_force * float(MVC) * 100.0)

    return thresholds
