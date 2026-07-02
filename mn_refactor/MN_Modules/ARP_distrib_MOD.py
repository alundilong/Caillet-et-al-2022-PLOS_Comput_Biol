"""Fit and complete the ARP distribution across identified MNs."""

from __future__ import annotations

import numpy as np

from PLOTS import plot_ARP_trendline_func
from Regression import regression


def ARP_distrib_func(test, Nb_MN, true_MN_pop, Real_MN_pop, saturating_MN, exp_ARP, muscle, plot, MN_pop):
    """Estimate ARP values for both saturating and non-saturating MNs."""
    n_mn = int(Nb_MN)
    real_pop = np.asarray(Real_MN_pop, dtype=float).reshape(-1)
    sat = np.asarray(saturating_MN, dtype=int).reshape(-1)
    exp = np.asarray(exp_ARP, dtype=float).reshape(-1)

    if sat.size > 4:
        def func_reg(x, a, b):
            return a * x**b

        x = real_pop[sat]
        y = exp
        popt, r2 = regression(x, y, "power", muscle, MN_pop)
        a_arp, b_arp = float(popt[0]), float(popt[1])
        if plot == "y":
            plot_ARP_trendline_func(x, y, true_MN_pop, func_reg, popt, a_arp, b_arp, r2)

        arp_table = np.asarray(func_reg(real_pop, *popt), dtype=float)
        arp_table[sat] = y
        non_saturating = np.setdiff1d(np.arange(n_mn), sat)
    else:
        a_arp = 0.0
        b_arp = 0.0
        set_arp = 1.0 / 20.0 if test == "GM_10" else 1.0 / 30.0
        arp_table = np.ones(n_mn, dtype=float) * set_arp
        non_saturating = np.array([], dtype=int)
        sat = np.array([], dtype=int)

    return a_arp, b_arp, arp_table, sat, non_saturating
