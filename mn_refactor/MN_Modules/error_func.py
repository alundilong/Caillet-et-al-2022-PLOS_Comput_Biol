"""Objective function for MN-size calibration."""

from __future__ import annotations

from FF_filt_func import FF_filt_func
from RMS_func import RMS_func


def error_func(Size, I, t_start, t_stop, t_plateau_end, Cm_rec, Cm_derec, step_size, ARP, kR, FIDF_exp):
    """RMS error between experimental and simulated FIDF for one MN size."""
    fidf_sim, _, _ = FF_filt_func(I, t_start, t_stop, t_plateau_end, Size, Cm_rec, Cm_derec, step_size, ARP, kR)
    return RMS_func(FIDF_exp, fidf_sim)
