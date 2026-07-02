"""Instantaneous discharge-frequency construction."""

from __future__ import annotations

import numpy as np

from _utils import clip_sample_indices, object_array


def _infer_samples(spikes, t_stop: float, fs: float) -> np.ndarray:
    """Accept spike times in seconds or samples and return sample indices."""
    arr = np.asarray(spikes, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr.astype(np.int64)
    # Original convention: values < 2*t_stop are assumed seconds.
    if arr[0] < 2.0 * float(t_stop):
        arr = arr * float(fs)
    return arr.astype(np.int64)


def _instantaneous_frequency_from_samples(samples: np.ndarray, fs: float) -> np.ndarray:
    if samples.size < 2:
        return np.array([], dtype=float)
    isi = np.diff(samples) / float(fs)
    valid = isi > 0
    out = np.zeros_like(isi, dtype=float)
    out[valid] = 1.0 / isi[valid]
    return out


def IDF_func(Nb_MN, time, disch_times, t_start, t_stop, fs=2048):
    """Compute instantaneous discharge frequencies and impulse trains.

    ``FF_FULL`` keeps the original meaning: a binary impulse is placed at each
    discharge time except the last discharge, so that subsequent Hanning
    filtering produces the FIDF signal.
    """
    n_mn = int(Nb_MN)
    n_time = len(time)
    fs = float(fs)

    if n_mn == 1:
        samples = _infer_samples(disch_times, t_stop=t_stop, fs=fs)
        idf = _instantaneous_frequency_from_samples(samples, fs=fs)
        ff_full = np.zeros(int(round(fs * (float(t_stop) - float(t_start)))), dtype=float)
        idx = samples[:-1] - int(round(float(t_start) * fs))
        idx = clip_sample_indices(idx, ff_full.size)
        ff_full[idx] = 1.0
        return idf, ff_full

    idf_dt = object_array(n_mn)
    ff_full = object_array(n_mn)
    for i in range(n_mn):
        samples = np.asarray(disch_times[i], dtype=float).ravel().astype(np.int64)
        idf_dt[i] = _instantaneous_frequency_from_samples(samples, fs=fs)
        impulse = np.zeros(n_time, dtype=float)
        idx = clip_sample_indices(samples[:-1], n_time)
        impulse[idx] = 1.0
        ff_full[i] = impulse
    return idf_dt, ff_full
