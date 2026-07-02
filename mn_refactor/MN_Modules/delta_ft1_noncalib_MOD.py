"""First-firing-time error for calibrated identified MNs."""

from __future__ import annotations

import numpy as np

from RC_LIF_MOD import RC_solve_func


def delta_ft1_noncalib_func(Nb_MN, I, t_start, t_stop, t_plateau_end, Calib_sizes, Cm_rec, Cm_derec, step_size, ARP_table, THRESHOLDS):
    """Return first-firing-time error for each identified MN."""
    n_mn = int(Nb_MN)
    first_times = np.zeros(n_mn, dtype=float)
    for i in range(n_mn):
        _, _, spikes, _ = RC_solve_func(I, t_start, t_stop, t_plateau_end, Calib_sizes[i], Cm_rec, Cm_derec, step_size, ARP_table[i])
        first_times[i] = spikes[0] if spikes.size else 0.0
    return first_times - np.asarray(THRESHOLDS[:n_mn, 0], dtype=float)
