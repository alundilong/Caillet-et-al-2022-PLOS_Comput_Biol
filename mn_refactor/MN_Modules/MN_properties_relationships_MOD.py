"""
Relationships between motoneuron properties used by the LIF model.

Original source: Caillet et al. 2022 PLOS Computational Biology codebase.
This file is included here because it is required by several modules but was
not present in the uploaded MN_Modules.zip.
"""

import numpy as np


def Ith_S_func(S):
    """Rheobase current [A] as a function of MN size S [m^2]."""
    return 3.82 * 10**8 * S**2.52


def S_Ith(Ith):
    """MN size S [m^2] as a function of rheobase current Ith [A]."""
    return 3.96 * 10**-4 * Ith**0.396


def R_S_func(S, kR=1.68e-10):
    """Membrane resistance [Ohm] as a function of MN size S [m^2]."""
    return kR / S**2.43


def C_S_func(S, Cm_rec):
    """Membrane capacitance [F] from specific capacitance and MN size."""
    return Cm_rec * S


def tau_R_C_func(R, C):
    """Membrane time constant [s]."""
    return R * C


def Fth_distrib_func(MN, muscle, MN_pop, k=1):
    """Typical muscle-specific MU force recruitment threshold distribution (%MVC)."""
    MN = np.asarray(MN)
    if muscle == "TA":
        if k == 2:
            return 1.11 * np.exp(0.045 * MN / 4)
        return 0.5052 * (58.1 * MN / MN_pop + 120 ** ((MN / MN_pop) ** 1.83))
    if muscle == "GM":
        return 0.6562 * (46.7 * MN / MN_pop + 90 ** ((MN / MN_pop) ** 1.79))
    raise ValueError(f"Unsupported muscle type: {muscle!r}")


def Ftet_norm_distrib_func(MN, muscle, MN_pop, k=1):
    """Normalized tetanic force distribution across the MU pool."""
    MN = np.asarray(MN)
    return 8.9324 * (3.0 * MN / MN_pop + 8.20 ** ((MN / MN_pop) ** 5.29))


def Ith_distrib_func(MN, true_MN_pop):
    """Typical rheobase current distribution [A] across a MN pool."""
    MN = np.asarray(MN)
    return 3.85e-9 * 9.1 ** ((MN / true_MN_pop) ** 1.1831)
