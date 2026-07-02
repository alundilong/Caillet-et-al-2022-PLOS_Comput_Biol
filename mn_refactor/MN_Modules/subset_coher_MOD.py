"""Coherence between random subsets of cumulative spike trains."""

import random
import numpy as np
from scipy import signal


def subset_coher_func(nb_tests, time, Nb_MN, Binary_matrix, fs=2048):
    """
    Derive average coherence in [1, 10] Hz between CSTs from random MN subsets.

    This version preserves the original random.sample behavior, but vectorizes
    summation of each subset.
    """
    nb_tests = int(nb_tests)
    Nb_MN = int(Nb_MN)
    mn_list = np.arange(Nb_MN)
    binary = np.asarray(Binary_matrix, dtype=float)
    avg_coher = np.ones(nb_tests, dtype=float)

    for k in range(nb_tests):
        set_1 = np.array(random.sample(range(Nb_MN), Nb_MN // 2), dtype=int)
        set_2 = np.setdiff1d(mn_list, set_1, assume_unique=False)
        cst1 = binary[set_1].sum(axis=0)
        cst2 = binary[set_2].sum(axis=0)
        _, cxy = signal.coherence(cst1, cst2, fs, nperseg=fs * 2, noverlap=fs)
        avg_coher[k] = np.mean(cxy[2:21])

    return float(np.average(avg_coher))
