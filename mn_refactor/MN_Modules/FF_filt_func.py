"""Simulate one MN and compute its filtered instantaneous discharge frequency."""

from __future__ import annotations

import numpy as np

from Hanning_filter_MOD import Hanning_filter_func
from IDF_MOD import IDF_func
from RC_LIF_MOD import RC_solve_func


def FF_filt_func(
    I,
    t_start,
    t_stop,
    t_plateau_end,
    Size,
    Cm_rec,
    Cm_derec,
    step_size,
    ARP,
    kR,
    adapt_kR="n",
    kR_derec=1.7e-10,
    fs=2048,
):
    """Return simulated FIDF, full impulse train, and discharge times."""
    _, _, disch_times, _ = RC_solve_func(
        I,
        t_start,
        t_stop,
        t_plateau_end,
        Size,
        Cm_rec,
        Cm_derec,
        step_size,
        ARP,
        kR,
        adapt_kR,
        kR_derec,
    )

    n_samples = int(round(float(fs) * (float(t_stop) - float(t_start))))
    if disch_times.size < 2:
        # Keep an obviously poor, correctly-sized signal for scalar minimization.
        fidf_sim = np.ones(n_samples, dtype=float)
        return fidf_sim, fidf_sim.copy(), disch_times

    time = np.arange(n_samples, dtype=float) / float(fs) + float(t_start)
    _, ff_full = IDF_func(1, time, disch_times, t_start, t_stop, fs=fs)
    fidf_sim = Hanning_filter_func(1, ff_full, fs=fs)
    return fidf_sim, ff_full, disch_times
