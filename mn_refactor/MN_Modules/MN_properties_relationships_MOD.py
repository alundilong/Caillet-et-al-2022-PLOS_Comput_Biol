"""Physiological relationships used by the motoneuron LIF model."""

from __future__ import annotations

import numpy as np


def Ith_S_func(S):
    """Rheobase current [A] as a function of MN surface area S [m^2]."""
    return 3.82e8 * np.asarray(S, dtype=float) ** 2.52


def S_Ith(Ith):
    """MN surface area S [m^2] as a function of rheobase current Ith [A]."""
    return 3.96e-4 * np.asarray(Ith, dtype=float) ** 0.396


def R_S_func(S, kR=1.68e-10):
    """Membrane resistance [Ohm] from MN size."""
    return float(kR) / np.asarray(S, dtype=float) ** 2.43


def C_S_func(S, Cm_rec):
    """Membrane capacitance [F] from specific capacitance and size."""
    return float(Cm_rec) * np.asarray(S, dtype=float)


def tau_R_C_func(R, C):
    """Membrane time constant [s]."""
    return np.asarray(R, dtype=float) * np.asarray(C, dtype=float)


def Fth_distrib_func(MN, muscle, MN_pop, k=1):
    """Muscle-specific force recruitment threshold distribution, in %MVC."""
    mn = np.asarray(MN, dtype=float)
    mn_pop = float(MN_pop)
    muscle = str(muscle)
    if muscle == "TA":
        if k == 2:
            return 1.11 * np.exp(0.045 * mn / 4.0)
        return 0.5052 * (58.1 * mn / mn_pop + 120.0 ** ((mn / mn_pop) ** 1.83))
    if muscle == "GM":
        return 0.6562 * (46.7 * mn / mn_pop + 90.0 ** ((mn / mn_pop) ** 1.79))
    raise ValueError(f"Unsupported muscle type: {muscle!r}")


def Ftet_norm_distrib_func(MN, muscle, MN_pop, k=1):
    """Normalized tetanic-force distribution across a MU pool."""
    mn = np.asarray(MN, dtype=float)
    return 8.9324 * (3.0 * mn / float(MN_pop) + 8.20 ** ((mn / float(MN_pop)) ** 5.29))


def Ith_distrib_func(MN, true_MN_pop):
    """Typical rheobase current distribution [A] across a MN pool."""
    mn = np.asarray(MN, dtype=float)
    return 3.85e-9 * 9.1 ** ((mn / float(true_MN_pop)) ** 1.1831)
