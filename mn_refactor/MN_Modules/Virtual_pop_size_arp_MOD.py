"""Reconstruct virtual MN size and ARP distributions."""

import numpy as np


def Virtual_size_arp_pop(MN_pop, a_size, c_size, exp_ARP, a_arp, b_arp):
    MN_pop = int(MN_pop)
    mn = np.arange(1, MN_pop + 1, dtype=float)
    virtual_size_arr = a_size * 2.4 ** ((mn / MN_pop) ** c_size)
    if len(exp_ARP) > 4:
        virtual_arp_arr = a_arp * mn ** b_arp
    else:
        virtual_arp_arr = np.ones(MN_pop, dtype=float) * 0.3
    # Return object arrays for maximum compatibility with the original code.
    return virtual_size_arr.astype(object), virtual_arp_arr.astype(object)
