"""Initial preprocessing of the force trace and simulation constants."""

from __future__ import annotations

import numpy as np


def preprocessing_func(author, Force, end_force, fs, Nb_MN, plateau_time1, plateau_time2):
    """Trim force, remove offset, and define common simulation parameters."""
    fs = int(fs)
    n_samples = int(float(end_force) * fs)

    force = np.asarray(Force, dtype=float).ravel()[:n_samples]
    if force.size < n_samples:
        raise ValueError(f"Force signal has {force.size} samples, expected at least {n_samples}.")

    baseline_samples = min(6000, force.size)
    force = force - np.min(force[:baseline_samples])
    time = np.arange(force.size, dtype=float) / fs
    mn_list = np.arange(int(Nb_MN), dtype=int)

    t_start = 0.0
    t_stop = float(end_force)
    t_stop_calib = (float(plateau_time1) + float(plateau_time2)) / 2.0
    kR = 1.68e-10
    Cm_rec = 1.3e-2
    step_size = 1e-4

    return force, time, mn_list, t_start, t_stop, t_stop_calib, kR, Cm_rec, step_size
