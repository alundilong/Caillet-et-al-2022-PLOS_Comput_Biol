"""Coherence analysis between random CST subsets."""

from __future__ import annotations

import random

import numpy as np
from scipy import signal


def subset_coher_func(nb_tests, time, Nb_MN, Binary_matrix, fs=2048):
    """Average coherence in the original [1, 10] Hz band approximation."""
    n_tests = int(nb_tests)
    n_mn = int(Nb_MN)
    binary = np.asarray(Binary_matrix, dtype=float)
    mn_ids = np.arange(n_mn)
    values = np.zeros(n_tests, dtype=float)

    for k in range(n_tests):
        set_1 = np.array(random.sample(range(n_mn), n_mn // 2), dtype=int)
        set_2 = np.setdiff1d(mn_ids, set_1, assume_unique=False)
        cst1 = binary[set_1].sum(axis=0)
        cst2 = binary[set_2].sum(axis=0)
        _, cxy = signal.coherence(cst1, cst2, fs=fs, nperseg=int(fs * 2), noverlap=int(fs))
        values[k] = np.mean(cxy[2:21])

    return float(np.mean(values))
