"""Load and order decomposed HDEMG motoneuron discharge data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io


def _extract_mat_cell_vector(cell_array):
    """Return a list of 1-D arrays from a MATLAB cell array."""
    arr = np.asarray(cell_array, dtype=object).squeeze()
    if arr.ndim == 0:
        return [np.asarray(arr.item()).squeeze()]
    return [np.asarray(arr.flat[i]).squeeze() for i in range(arr.size)]


def _clean_spike_samples(entry) -> np.ndarray:
    values = np.asarray(entry, dtype=float).ravel()
    values = values[np.isfinite(values)]
    return values.astype(np.int64)


def EXP_DATA_PROCESSING_func(author, test):
    """Load ``Input_Exp_Data/<test>.mat`` and sort MUs by first discharge.

    The ``author`` argument is preserved for API compatibility; the uploaded
    datasets already encode the required information in their file names.
    """
    data_path = Path("Input_Exp_Data") / f"{test}.mat"
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find experimental data file: {data_path}")

    mat = scipy.io.loadmat(data_path)
    if "MUPulses" not in mat or "ref_signal" not in mat:
        raise KeyError("The .mat file must contain 'MUPulses' and 'ref_signal'.")

    force = np.asarray(mat["ref_signal"], dtype=float).squeeze()
    raw_cells = _extract_mat_cell_vector(mat["MUPulses"])
    discharge_times = [_clean_spike_samples(cell) for cell in raw_cells]
    discharge_times = [dt for dt in discharge_times if dt.size > 0]
    if not discharge_times:
        raise ValueError(f"No non-empty discharge trains found in {data_path}")

    first_discharge = np.array([dt[0] for dt in discharge_times], dtype=float)
    order = np.argsort(first_discharge)

    sorted_discharge_times = np.empty((len(order),), dtype=object)
    for out_idx, raw_idx in enumerate(order):
        sorted_discharge_times[out_idx] = discharge_times[raw_idx]

    return len(sorted_discharge_times), force, sorted_discharge_times
