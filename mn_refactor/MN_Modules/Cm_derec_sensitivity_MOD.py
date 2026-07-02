"""Sensitivity analysis for derecruitment capacitance Cm_derec."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from FF_filt_func import FF_filt_func
from RMS_func import RMS_func
from _utils import safe_corrcoef


def Cm_derec_sensitivity_func(
    Nb_MN,
    time,
    I,
    Cm_rec,
    Cm_derec_array,
    Calib_sizes_final,
    ARP_table,
    FIDF_exp,
    plateau_time1,
    end_force,
    step_size,
    kR,
    adapt_kR="n",
    kR_derec=1.7e-10,
    fs=2048,
):
    """Evaluate derecruitment FIDF fit over candidate Cm_derec values."""
    n_mn = int(Nb_MN)
    t1 = float(plateau_time1)
    t2 = float(end_force)
    range_t1 = int(round(t1 * fs))
    range_t2 = int(round(t2 * fs))

    rms_by_cm = np.zeros(len(Cm_derec_array), dtype=float)
    r2_by_cm = np.zeros(len(Cm_derec_array), dtype=float)

    for j, Cm_derec in enumerate(np.asarray(Cm_derec_array, dtype=float)):
        rms_table = np.zeros(n_mn, dtype=float)
        r2_table = np.zeros(n_mn, dtype=float)
        for i in range(n_mn):
            if i % 5 == 0:
                print(f"Firing MN n° {i} with Cm_derec = {Cm_derec:.4g}")
            fidf_sim, _, _ = FF_filt_func(
                I,
                t1,
                t2,
                t1,
                Calib_sizes_final[i],
                Cm_rec,
                Cm_derec,
                step_size,
                ARP_table[i],
                kR,
                adapt_kR,
                kR_derec,
                fs=fs,
            )
            ff_exp = np.asarray(FIDF_exp[i][range_t1:range_t2], dtype=float)
            n = min(ff_exp.size, fidf_sim.size)
            ff_exp = ff_exp[:n]
            fidf_sim = fidf_sim[:n]
            denom = np.max(ff_exp) if np.max(ff_exp) > 0 else 1.0
            rms_table[i] = RMS_func(ff_exp, fidf_sim) / denom * 100.0
            r2_table[i] = safe_corrcoef(fidf_sim, ff_exp)

            if i % 4 == 0:
                plot_time = np.asarray(time[range_t1 : range_t1 + n], dtype=float)
                plt.plot(plot_time, ff_exp, "k", label="Experimental")
                plt.plot(plot_time, fidf_sim, label="Simulated")
                plt.xlabel("Time(s)")
                plt.ylabel("Filtered discharge intervals")
                plt.title(f"MN {i}")
                plt.grid()
                plt.legend()
                plt.show()

        rms_by_cm[j] = float(np.mean(rms_table))
        r2_by_cm[j] = float(np.mean(r2_table))

    normalized_rms = rms_by_cm / np.max(rms_by_cm) if np.max(rms_by_cm) > 0 else rms_by_cm
    best_idx = int(np.argmin(normalized_rms - r2_by_cm))
    return np.asarray(Cm_derec_array, dtype=float), rms_by_cm, r2_by_cm, float(Cm_derec_array[best_idx])
