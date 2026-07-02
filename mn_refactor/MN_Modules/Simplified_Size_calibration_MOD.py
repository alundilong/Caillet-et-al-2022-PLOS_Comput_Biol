"""Batch calibration of MN size from experimental FIDF traces."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from error_func import error_func
from FF_filt_func import FF_filt_func
from MN_properties_relationships_MOD import R_S_func
from PLOTS import plot_exp_pred_FIDFs_func
from _utils import safe_corrcoef


def _calibrate_single_mn(
    i,
    I,
    t_start,
    t_stop,
    plateau_time2,
    size_min,
    size_max,
    step_size,
    kR,
    Cm_rec,
    ff_filt_exp,
    arp,
):
    sol = minimize_scalar(
        error_func,
        bounds=(size_min, size_max),
        args=(I, t_start, t_stop, plateau_time2, Cm_rec, Cm_rec, step_size, arp, kR, ff_filt_exp),
        method="bounded",
        options={"xatol": 1e-9},
    )
    fidf_sim, _, _ = FF_filt_func(I, t_start, t_stop, plateau_time2, sol.x, Cm_rec, Cm_rec, step_size, arp, kR)
    return sol, fidf_sim


def Size_calibration_function(
    Cm_rec,
    time,
    t_start,
    t_stop,
    plateau_time2,
    end_force,
    Size_min,
    Size_max,
    step_size,
    kR,
    Nb_MN,
    FIDF_exp,
    ARP_table,
    I,
    plot="y",
    fs=2048,
):
    """Calibrate each identified MN size by minimizing FIDF RMS error."""
    n_mn = int(Nb_MN)
    range_start = int(round(float(t_start) * fs))
    range_stop = int(round(float(t_stop) * fs))

    rms_table = np.empty((n_mn,), dtype=object)
    corrcoef_table = np.empty((n_mn,), dtype=object)
    calib_sizes = np.empty((n_mn,), dtype=object)

    for i in range(n_mn):
        print(f"Calibrating size of MN n° {i}")
        ff_filt_exp = np.asarray(FIDF_exp[i][range_start:range_stop], dtype=float)
        arp = float(np.asarray(ARP_table[i], dtype=float).ravel()[0])
        sol, fidf_sim = _calibrate_single_mn(
            i,
            I,
            t_start,
            t_stop,
            plateau_time2,
            float(Size_min),
            float(Size_max),
            step_size,
            kR,
            Cm_rec,
            ff_filt_exp,
            arp,
        )

        if fidf_sim.size > 1 and np.max(ff_filt_exp) > 0:
            rms_table[i] = float(sol.fun / np.max(ff_filt_exp) * 100.0)
            corrcoef_table[i] = safe_corrcoef(fidf_sim, ff_filt_exp)
            calib_sizes[i] = float(sol.x)
        else:
            # Preserve original fallback behavior for non-firing candidate MUs.
            rms_table[i] = rms_table[i - 1] if i > 0 else np.nan
            corrcoef_table[i] = corrcoef_table[i - 1] if i > 0 else 0.0
            calib_sizes[i] = calib_sizes[i - 1] if i > 0 else float(sol.x)

        if plot == "y":
            plot_exp_pred_FIDFs_func(
                time,
                range_start,
                range_stop,
                ff_filt_exp,
                I,
                t_start,
                t_stop,
                plateau_time2,
                float(sol.x),
                Cm_rec,
                Cm_rec,
                step_size,
                arp,
                i,
                kR,
            )

    r_table = R_S_func(calib_sizes.astype(float), kR)
    return calib_sizes, r_table, rms_table, corrcoef_table
