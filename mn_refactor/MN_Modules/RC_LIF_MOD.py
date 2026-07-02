"""Leaky integrate-and-fire motoneuron model."""

from __future__ import annotations

import numpy as np

from MN_properties_relationships_MOD import C_S_func, R_S_func, tau_R_C_func
from _utils import get_current_values


def RC_solve_func(
    I,
    t_start,
    t_stop,
    t_plateau_end,
    Size,
    Cm_rec,
    Cm_derec,
    step_size,
    ARP,
    kR=1.68e-10,
    adapt_kR="n",
    kR_derec=1.7e-10,
    ARP_rand=10,
):
    """Solve the RC LIF model for one motoneuron.

    This is a drop-in replacement for the original solver.  The update equation
    and random ARP perturbation are preserved, but repeated ``np.append`` calls
    are replaced with a Python list and the current input is pre-evaluated when
    possible.
    """
    t_start = float(t_start)
    t_stop = float(t_stop)
    t_plateau_end = float(t_plateau_end)
    step_size = float(step_size)
    size = float(Size)
    arp_initial = float(np.asarray(ARP, dtype=float).ravel()[0])

    Vth = 27e-3
    R_rec = float(R_S_func(size, kR))
    C_rec = float(C_S_func(size, Cm_rec))
    tau_rec = float(tau_R_C_func(R_rec, C_rec))

    if adapt_kR == "y":
        R_derec = float(R_S_func(size, kR_derec))
    else:
        R_derec = R_rec
    C_derec = float(C_S_func(size, Cm_derec))
    tau_derec = float(tau_R_C_func(R_derec, C_derec))

    parameters = [Vth, arp_initial, R_rec, C_rec, tau_rec]
    tim_list = np.arange(t_start, t_stop, step_size)
    if tim_list.size == 0:
        return tim_list, np.array([], dtype=float), np.array([], dtype=float), parameters

    I_values = get_current_values(I, tim_list)
    V = np.zeros(tim_list.size, dtype=float)
    firing_times = []

    Vnt = R_rec * step_size / tau_rec * I_values[0]
    t_fire = -7.0 * tau_rec
    arp = arp_initial

    for idx, nt in enumerate(tim_list):
        if idx == 0:
            V[idx] = Vnt
            continue

        if nt > t_plateau_end:
            R = R_derec
            tau = tau_derec
        else:
            R = R_rec
            tau = tau_rec

        Vnt = np.exp(-step_size / tau) * Vnt + R * step_size / tau * I_values[idx]

        if Vnt > Vth:
            Vnt = 0.0
            V[idx] = 0.0
            firing_times.append(nt)
            t_fire = nt
            if arp_initial > 0:
                arp = float(np.random.normal(arp_initial, arp_initial / float(ARP_rand)))
            else:
                arp = 0.0
        elif nt > t_fire and nt < t_fire + arp:
            Vnt = 0.0
            V[idx] = 0.0
        else:
            V[idx] = Vnt

    return tim_list, V, np.asarray(firing_times, dtype=float), parameters
