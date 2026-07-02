"""Map identified MNs into the literature-based full MN pool."""

from __future__ import annotations

import numpy as np

from MN_properties_relationships_MOD import Fth_distrib_func


def MN_distirbution_func(Nb_MN, true_MN_pop, MN_pool_list, muscle, THRESHOLDS):
    """Return nearest full-pool index for each experimental recruitment threshold."""
    model_thresholds = np.asarray(Fth_distrib_func(MN_pool_list, muscle, true_MN_pop), dtype=float)
    exp_thresholds = np.asarray(THRESHOLDS[: int(Nb_MN), 2], dtype=float)
    return np.abs(exp_thresholds[:, None] - model_thresholds[None, :]).argmin(axis=1).astype(int)
