"""Experimental inert-period / ARP estimation."""

from __future__ import annotations

import numpy as np


def _poly_trend(x: np.ndarray, y: np.ndarray, degree: int = 12) -> np.ndarray:
    """Polynomial trend with automatic degree reduction for short vectors."""
    if x.size == 0:
        return np.array([], dtype=float)
    deg = min(int(degree), max(0, x.size - 1))
    if deg == 0:
        return np.ones_like(x, dtype=float) * float(np.mean(y))
    return np.poly1d(np.polyfit(x.astype(float), y.astype(float), deg))(x.astype(float))


def exp_ARP_func(Nb_MN, plateau_time1, plateau_time2, end_force, disch_times, IDF_dt, fs=2048):
    """Identify saturating MNs and infer ARP as 1 / max trend firing rate."""
    n_mn = int(Nb_MN)
    fs = float(fs)
    plateau_time1 = float(plateau_time1)
    plateau_time2 = float(plateau_time2)

    ff_trend = np.empty((n_mn,), dtype=object)
    ff_trend_max = np.zeros(n_mn, dtype=float)
    ff_plateau_mean = np.zeros(n_mn, dtype=float)

    for mn in range(n_mn):
        spikes = np.asarray(disch_times[mn], dtype=float).ravel()
        x = spikes[:-1] / fs
        y = np.asarray(IDF_dt[mn], dtype=float).ravel()
        n = min(x.size, y.size)
        x = x[:n]
        y = y[:n]
        trend = _poly_trend(x, y, degree=12)
        ff_trend[mn] = trend
        ff_trend_max[mn] = np.max(trend) if trend.size else np.nan
        plateau_mask = (x > plateau_time1) & (x < plateau_time2)
        ff_plateau_mean[mn] = np.mean(trend[plateau_mask]) if np.any(plateau_mask) else np.nan

    saturation_times = np.zeros(n_mn, dtype=float)
    for mn in range(n_mn):
        spikes = np.asarray(disch_times[mn], dtype=float).ravel()
        before_plateau = spikes[spikes < (plateau_time1 - 1.0) * fs] / fs
        if before_plateau.size == 0 or not np.isfinite(ff_plateau_mean[mn]):
            continue
        trend_before = np.asarray(ff_trend[mn], dtype=float)[: before_plateau.size]
        mean_threshold = ff_plateau_mean[mn] * 0.95
        above = np.flatnonzero(trend_before > mean_threshold)
        if above.size == 0 or trend_before[0] > mean_threshold:
            continue
        idx = max(0, int(above[0]) - 1)
        saturation_times[mn] = spikes[idx] / fs

    saturating_mn = np.argwhere(saturation_times > 0)
    if saturating_mn.size == 0:
        exp_arp = np.empty((0, 1), dtype=float)
    else:
        exp_arp = 1.0 / ff_trend_max[saturating_mn]
    return saturating_mn, exp_arp
