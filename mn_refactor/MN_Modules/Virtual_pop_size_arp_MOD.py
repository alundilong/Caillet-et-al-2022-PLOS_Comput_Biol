"""Construct virtual MN size and ARP arrays."""

from __future__ import annotations

import numpy as np


def Virtual_size_arp_pop(MN_pop, a_size, c_size, exp_ARP, a_arp, b_arp):
    """Return virtual size and ARP distributions for the complete MN pool."""
    mn_pop = int(MN_pop)
    mn = np.arange(1, mn_pop + 1, dtype=float)
    virtual_size_arr = float(a_size) * 2.4 ** ((mn / mn_pop) ** float(c_size))
    if len(exp_ARP) > 4:
        virtual_arp_arr = float(a_arp) * mn ** float(b_arp)
    else:
        # Kept compatible with the original uploaded code.
        virtual_arp_arr = np.ones(mn_pop, dtype=float) * 0.3
    return virtual_size_arr.astype(object), virtual_arp_arr.astype(object)
