"""Map experimentally identified MNs into a real MN pool."""

import numpy as np
from MN_properties_relationships_MOD import Fth_distrib_func


def MN_distirbution_func(Nb_MN, true_MN_pop, MN_pool_list, muscle, THRESHOLDS):
    """
    Compute MN locations by nearest match to literature recruitment thresholds.

    Vectorized replacement for the original loop. Preserves returned indices.
    """
    model_thresholds = Fth_distrib_func(MN_pool_list, muscle, true_MN_pop)
    exp_thresholds = np.asarray(THRESHOLDS[: int(Nb_MN), 2], dtype=float)
    diff = np.abs(model_thresholds[None, :] - exp_thresholds[:, None])
    return diff.argmin(axis=1).astype(int)
