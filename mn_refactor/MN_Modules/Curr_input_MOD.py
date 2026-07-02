"""Map experimental common input to model synaptic current input."""

from __future__ import annotations

import numpy as np

from MN_properties_relationships_MOD import Ith_distrib_func


def Common_to_current_input_func(THRESHOLDS, Real_MN_pop, true_MN_pop, common_input, fs=2048):
    """Return the affine gain ``G`` and intercept ``I1`` for I(t).

    The mapping follows the original code:

    ``I(t) = I1 + G * common_input(t)`` while firing activity is present.
    ``I1`` and the last-current value are inferred from the literature-based
    rheobase distribution at the first and last identified MN locations.
    """
    thresholds = np.asarray(THRESHOLDS, dtype=float)
    real_pop = np.asarray(Real_MN_pop, dtype=float).ravel()
    ci = np.asarray(common_input, dtype=float).ravel()
    max_ci = float(np.max(ci))
    if max_ci == 0:
        raise ValueError("common_input is all zeros; cannot derive current input.")

    I1 = float(Ith_distrib_func(real_pop[0], true_MN_pop))
    Ilast = float(Ith_distrib_func(real_pop[-1], true_MN_pop))
    CI1 = thresholds[0, 1] / 100.0 * max_ci
    CIlast = thresholds[-1, 1] / 100.0 * max_ci
    if np.isclose(CIlast, CI1):
        raise ZeroDivisionError("First and last CI thresholds are identical; cannot compute G.")

    G = (Ilast - I1) / (CIlast - CI1)
    return float(G), float(I1)
