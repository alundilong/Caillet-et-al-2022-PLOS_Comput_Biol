"""Final LIF simulation of the complete virtual MN pool."""

from __future__ import annotations

import numpy as np

from RC_LIF_MOD import RC_solve_func


def run_LIF_simulation_func(
    Nb_MU,
    MN_pop,
    Real_MU_pop,
    time,
    t_start,
    t_stop,
    plateau_time2,
    FF_FULL,
    FF_filt_arr,
    Virtual_size_arr,
    Virtual_ARP_arr,
    I,
    Cm_rec,
    Cm_derec,
    step_size,
    kR,
    plot,
    adapt_kR="n",
    kR_derec=1.7e-10,
    fs=2048,
):
    """Simulate firing times for every MN in the virtual population."""
    mn_pop = int(MN_pop)
    firing_times_sim = np.empty((mn_pop,), dtype=object)
    rms_table_sim = np.empty((int(Nb_MU),), dtype=object)
    nme_table_sim = np.empty((int(Nb_MU),), dtype=object)
    corrcoef_table_sim = np.empty((int(Nb_MU),), dtype=object)

    for mn in range(mn_pop):
        if (mn + 1) % 10 == 0:
            print(f"Simulating MN n° {mn + 1}")
        _, _, firing_times, _ = RC_solve_func(
            I,
            t_start,
            t_stop,
            plateau_time2,
            Virtual_size_arr[mn],
            Cm_rec,
            Cm_derec,
            step_size,
            Virtual_ARP_arr[mn],
            kR,
            adapt_kR,
            kR_derec,
        )
        firing_times_sim[mn] = firing_times

    return nme_table_sim, rms_table_sim, corrcoef_table_sim, firing_times_sim
