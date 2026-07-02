#!/usr/bin/env python3
"""
Visualize experimental motoneuron / motor-unit discharge data for all cases.

Expected project layout
-----------------------
project_root/
├── Input_Exp_Data/
│   ├── GM_30.mat
│   ├── TA_35_D.mat
│   ├── TA_35_H.mat
│   └── TA_50.mat
└── visualize_all_exp_data.py

The .mat files are expected to contain at least:
    - ref_signal : reference force/torque-like signal
    - MUPulses   : MATLAB cell array, one cell per MU/MN, containing spike sample indices
    - fsamp      : sampling frequency, optional. If absent, metadata default is used.

This script creates, for each dataset:
    1. Reference signal
    2. MU firing raster sorted by recruitment time
    3. Smoothed cumulative spike train vs normalized reference
    4. Instantaneous firing rates
    5. Recruitment/derecruitment reference values

It also creates:
    - overview_all_cases.png
    - summary_all_cases.csv

Notes
-----
The original Caillet et al. code treats discharge times as sample indices at fs=2048 Hz.
This script keeps that behavior by default with --index-base original.
Use --index-base one if your MUPulses are MATLAB 1-based sample indices.
Use --index-base auto to subtract 1 only when an index would otherwise exceed signal length.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Dataset metadata copied from the original model scripts.
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CaseInfo:
    author: str
    dataset: str
    muscle: str
    plateau_time1: float
    plateau_time2: float
    end_force: float
    mvc_fraction: float
    mn_pool: int
    default_fs: float

    @property
    def mvc_percent(self) -> float:
        return self.mvc_fraction * 100.0


CASE_TABLE: Dict[str, CaseInfo] = {
    "TA_35_D": CaseInfo("DelVecchio", "TA_35_D", "TA", 10.3, 20.5, 30.0, 0.35, 400, 2048.0),
    "TA_35_H": CaseInfo("Hug",       "TA_35_H", "TA", 10.5, 20.5, 30.0, 0.35, 400, 2048.0),
    "TA_50":   CaseInfo("Hug",       "TA_50",   "TA", 12.0, 21.8, 34.5, 0.50, 400, 2048.0),
    "GM_30":   CaseInfo("Hug",       "GM_30",   "GM", 9.1,  19.1, 33.5, 0.30, 550, 2048.0),
}

DEFAULT_CASES = ("TA_35_D", "TA_35_H", "TA_50", "GM_30")


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class ExpCaseData:
    info: CaseInfo
    fs: float
    time: np.ndarray
    ref_signal: np.ndarray
    disch_times: np.ndarray          # object array, sorted by first discharge time
    original_mu_ids: np.ndarray      # original MU IDs after sorting
    binary_matrix: np.ndarray        # shape: n_mus x n_samples
    cst: np.ndarray
    common_input_10hz: np.ndarray
    common_control_4hz: np.ndarray
    smoothed_cst: np.ndarray
    smoothed_cst_norm: np.ndarray
    ref_norm: np.ndarray
    first_spike_idx: np.ndarray
    last_spike_idx: np.ndarray

    @property
    def n_mus(self) -> int:
        return len(self.disch_times)

    @property
    def duration(self) -> float:
        return float(self.time[-1]) if len(self.time) else 0.0


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def normalize_minmax(x: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1]. Returns zeros if signal is constant."""
    x = np.asarray(x, dtype=float)
    y = x - np.nanmin(x)
    denom = np.nanmax(y)
    if denom > 0:
        y = y / denom
    return y


def normalize_by_max(x: np.ndarray) -> np.ndarray:
    """Divide by max absolute positive amplitude; safe for all-zero vectors."""
    x = np.asarray(x, dtype=float)
    denom = np.nanmax(np.abs(x))
    return x / denom if denom > 0 else np.zeros_like(x, dtype=float)


def lowpass_filter(signal: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter."""
    signal = np.asarray(signal, dtype=float)
    nyq = 0.5 * fs
    cutoff = cutoff_hz / nyq
    cutoff = min(max(cutoff, 1e-6), 0.999999)
    b, a = butter(order, cutoff, btype="low")

    # filtfilt requires enough samples. If not enough, return raw signal.
    min_len = 3 * max(len(a), len(b))
    if signal.size <= min_len:
        return signal.copy()
    return filtfilt(b, a, signal)


def extract_scalar(mat: dict, key: str, default: Optional[float] = None) -> float:
    if key not in mat:
        if default is None:
            raise KeyError(f"Required key not found in .mat file: {key}")
        return float(default)
    return float(np.asarray(mat[key]).squeeze())


def extract_ref_signal(mat: dict) -> np.ndarray:
    if "ref_signal" not in mat:
        raise KeyError("Required variable 'ref_signal' was not found in .mat file.")
    return np.asarray(mat["ref_signal"]).squeeze().astype(float)


def extract_mu_cells(mupulses_raw: np.ndarray) -> List[np.ndarray]:
    """
    Extract one 1D spike-index vector per MU from scipy-loaded MATLAB MUPulses.

    In the original code:
        disch_times_raw = np.array(value)[0]
        disch_times_raw[i][0]
    This function is more robust but keeps the same interpretation.
    """
    raw = np.asarray(mupulses_raw)

    if raw.dtype == object:
        # Usually shape is (1, n_mus). Squeeze to length n_mus.
        cells = raw.squeeze()
        if cells.ndim == 0:
            cells = np.array([cells.item()], dtype=object)

        out: List[np.ndarray] = []
        for item in cells.flat:
            arr = np.asarray(item).squeeze()
            out.append(np.asarray(arr).flatten())
        return out

    # Fallback: numeric matrix, either rows or columns are MUs.
    if raw.ndim == 1:
        return [raw.flatten()]
    if raw.ndim == 2:
        if raw.shape[0] <= raw.shape[1]:
            return [raw[i, :].flatten() for i in range(raw.shape[0])]
        return [raw[:, i].flatten() for i in range(raw.shape[1])]

    raise ValueError(f"Unsupported MUPulses array shape: {raw.shape}, dtype={raw.dtype}")


def clean_spike_indices(
    x: np.ndarray,
    n_samples: int,
    index_base: str = "original",
) -> np.ndarray:
    """
    Clean spike sample indices.

    Parameters
    ----------
    x:
        Raw spike index vector from one MUPulses cell.
    n_samples:
        Length of reference signal.
    index_base:
        'original': do not shift, matching the original Caillet scripts.
        'zero':     do not shift.
        'one':      subtract 1 from all positive sample indices.
        'auto':     subtract 1 only if max index >= n_samples.
    """
    x = np.asarray(x).squeeze()
    if x.size == 0:
        return np.array([], dtype=np.int64)

    if np.issubdtype(x.dtype, np.number):
        x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([], dtype=np.int64)

    idx = np.rint(x.astype(float)).astype(np.int64)

    if index_base not in {"original", "zero", "one", "auto"}:
        raise ValueError("index_base must be one of: original, zero, one, auto")

    if index_base == "one":
        idx = idx - 1
    elif index_base == "auto":
        # Keep original behavior unless an index would be out of bounds.
        if idx.size > 0 and np.nanmax(idx) >= n_samples:
            idx = idx - 1
    # 'original' and 'zero' do nothing.

    idx = idx[(idx >= 0) & (idx < n_samples)]
    idx = np.unique(idx)
    return idx.astype(np.int64)


def sort_mus_by_recruitment(spike_lists: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort spike trains by first discharge time. Returns sorted object array and original IDs.
    """
    first = np.array([sp[0] if len(sp) else np.inf for sp in spike_lists], dtype=float)
    order = np.argsort(first)
    sorted_lists = np.empty(len(spike_lists), dtype=object)
    for rank, idx in enumerate(order):
        sorted_lists[rank] = spike_lists[idx]
    return sorted_lists, order + 1  # 1-based original MU IDs for display


def build_binary_matrix(spike_lists: Sequence[np.ndarray], n_samples: int) -> np.ndarray:
    """Build n_mus x n_samples binary spike matrix."""
    binary = np.zeros((len(spike_lists), n_samples), dtype=np.uint8)
    for i, sp in enumerate(spike_lists):
        if len(sp):
            binary[i, sp] = 1
    return binary


def instantaneous_firing_rates(spike_idx: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return times and instantaneous firing rates from spike sample indices."""
    if len(spike_idx) < 2:
        return np.array([]), np.array([])
    isi = np.diff(spike_idx) / fs
    valid = isi > 0
    if not np.any(valid):
        return np.array([]), np.array([])
    rates = 1.0 / isi[valid]
    times = spike_idx[1:][valid] / fs
    return times, rates


def load_case(
    input_dir: Path,
    case_name: str,
    index_base: str,
    smoothing_sigma_s: float,
) -> ExpCaseData:
    """Load and process one .mat dataset."""
    if case_name not in CASE_TABLE:
        raise KeyError(f"Unknown case '{case_name}'. Known cases: {list(CASE_TABLE)}")
    info = CASE_TABLE[case_name]
    mat_path = input_dir / f"{case_name}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Cannot find {mat_path}")

    mat = scipy.io.loadmat(mat_path)
    fs = extract_scalar(mat, "fsamp", default=info.default_fs)
    ref_signal = extract_ref_signal(mat)
    n_samples = len(ref_signal)
    time = np.arange(n_samples) / fs

    if "MUPulses" not in mat:
        raise KeyError(f"Required variable 'MUPulses' not found in {mat_path}")

    raw_cells = extract_mu_cells(mat["MUPulses"])
    spike_lists_raw = [clean_spike_indices(x, n_samples, index_base=index_base) for x in raw_cells]
    disch_times, original_mu_ids = sort_mus_by_recruitment(spike_lists_raw)
    binary_matrix = build_binary_matrix(disch_times, n_samples)
    cst = binary_matrix.sum(axis=0).astype(float)

    common_input_10hz = lowpass_filter(cst, 10.0, fs)
    common_control_4hz = lowpass_filter(cst, 4.0, fs)

    sigma_samples = max(float(smoothing_sigma_s) * fs, 1.0)
    smoothed_cst = gaussian_filter1d(cst, sigma=sigma_samples)
    smoothed_cst_norm = normalize_minmax(smoothed_cst)
    ref_norm = normalize_minmax(ref_signal)

    first_spike_idx = np.array([sp[0] if len(sp) else -1 for sp in disch_times], dtype=int)
    last_spike_idx = np.array([sp[-1] if len(sp) else -1 for sp in disch_times], dtype=int)

    return ExpCaseData(
        info=info,
        fs=fs,
        time=time,
        ref_signal=ref_signal,
        disch_times=disch_times,
        original_mu_ids=original_mu_ids,
        binary_matrix=binary_matrix,
        cst=cst,
        common_input_10hz=common_input_10hz,
        common_control_4hz=common_control_4hz,
        smoothed_cst=smoothed_cst,
        smoothed_cst_norm=smoothed_cst_norm,
        ref_norm=ref_norm,
        first_spike_idx=first_spike_idx,
        last_spike_idx=last_spike_idx,
    )


def summarize_case(case: ExpCaseData) -> dict:
    """Return summary statistics for CSV/terminal."""
    spike_counts = case.binary_matrix.sum(axis=1).astype(int)
    total_spikes = int(spike_counts.sum())
    duration = len(case.time) / case.fs
    mean_fr_all = total_spikes / case.n_mus / duration if duration > 0 and case.n_mus > 0 else np.nan

    active_duration_rates = []
    median_inst_rates = []
    first_times = []
    last_times = []
    rec_ref_norm = []
    derec_ref_norm = []

    peak_ref = np.nanmax(case.ref_signal)
    ref_range = np.nanmax(case.ref_signal) - np.nanmin(case.ref_signal)

    for sp in case.disch_times:
        if len(sp) > 0:
            first_times.append(sp[0] / case.fs)
            last_times.append(sp[-1] / case.fs)
            rec_ref_norm.append(case.ref_norm[sp[0]])
            derec_ref_norm.append(case.ref_norm[sp[-1]])
        if len(sp) > 1:
            active_duration = (sp[-1] - sp[0]) / case.fs
            if active_duration > 0:
                active_duration_rates.append((len(sp) - 1) / active_duration)
            _, rates = instantaneous_firing_rates(sp, case.fs)
            if len(rates):
                median_inst_rates.append(np.nanmedian(rates))

    return {
        "dataset": case.info.dataset,
        "author": case.info.author,
        "muscle": case.info.muscle,
        "target_mvc_percent": case.info.mvc_percent,
        "fs_hz": case.fs,
        "duration_s": duration,
        "n_mus": case.n_mus,
        "total_spikes": total_spikes,
        "mean_fr_all_duration_hz_per_mu": mean_fr_all,
        "mean_fr_active_duration_hz": float(np.nanmean(active_duration_rates)) if active_duration_rates else np.nan,
        "median_instantaneous_fr_hz": float(np.nanmedian(median_inst_rates)) if median_inst_rates else np.nan,
        "first_identified_spike_s": float(np.nanmin(first_times)) if first_times else np.nan,
        "last_identified_spike_s": float(np.nanmax(last_times)) if last_times else np.nan,
        "ref_min": float(np.nanmin(case.ref_signal)),
        "ref_max": float(np.nanmax(case.ref_signal)),
        "ref_range": float(ref_range),
        "mean_recruitment_ref_norm": float(np.nanmean(rec_ref_norm)) if rec_ref_norm else np.nan,
        "mean_derecruitment_ref_norm": float(np.nanmean(derec_ref_norm)) if derec_ref_norm else np.nan,
    }


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_case(case: ExpCaseData, output_dir: Path, show: bool = False) -> Path:
    """Create a multi-panel diagnostic figure for one dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)

    info = case.info
    time = case.time
    fs = case.fs
    ref = case.ref_signal
    n_mus = case.n_mus

    fig, axes = plt.subplots(5, 1, figsize=(15, 14), sharex=False)
    fig.suptitle(
        f"{info.dataset}: {info.muscle}, {info.mvc_percent:.0f}% MVC target, "
        f"{n_mus} identified MUs/MNs",
        fontsize=15,
        fontweight="bold",
    )

    # Panel 1: Reference signal
    ax = axes[0]
    ax.plot(time, ref, color="black", linewidth=1.1)
    ax.axvspan(info.plateau_time1, info.plateau_time2, alpha=0.12, label="plateau window")
    ax.set_title("Reference signal")
    ax.set_ylabel("Ref signal")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(time[0], time[-1])

    # Panel 2: Raster
    ax = axes[1]
    spike_total = 0
    for rank, sp in enumerate(case.disch_times, start=1):
        if len(sp):
            spike_times = sp / fs
            ax.scatter(
                spike_times,
                np.full_like(spike_times, rank, dtype=float),
                marker="|",
                s=7,
                linewidths=0.7,
                color="black",
            )
            spike_total += len(sp)
    ax.set_title(f"MU/MN firing raster, sorted by first discharge ({spike_total} spikes)")
    ax.set_ylabel("Recruitment rank")
    ax.set_ylim(0.5, n_mus + 0.5)
    ax.grid(True, alpha=0.25, axis="x")
    ax.set_xlim(time[0], time[-1])

    # Panel 3: CST/common drive vs reference
    ax = axes[2]
    ax.plot(time, case.ref_norm, color="black", linewidth=1.2, label="normalized reference")
    ax.plot(time, normalize_minmax(case.common_input_10hz), linewidth=1.0, label="CST low-pass 10 Hz")
    ax.plot(time, normalize_minmax(case.common_control_4hz), linewidth=1.2, label="CST low-pass 4 Hz")
    ax.plot(time, case.smoothed_cst_norm, linewidth=0.9, alpha=0.75, label="Gaussian-smoothed CST")
    ax.set_title("Normalized reference vs smoothed cumulative spike train")
    ax.set_ylabel("Normalized")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(time[0], time[-1])

    # Panel 4: Instantaneous firing rates
    ax = axes[3]
    for rank, sp in enumerate(case.disch_times, start=1):
        t_fr, rates = instantaneous_firing_rates(sp, fs)
        if len(rates):
            ax.plot(t_fr, rates, linewidth=0.7, alpha=0.75)
    ax.axhspan(5, 35, alpha=0.08, label="typical voluntary range, approx. 5-35 Hz")
    ax.set_title("Instantaneous firing rates")
    ax.set_ylabel("Hz")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(time[0], time[-1])

    # Panel 5: Recruitment/derecruitment reference values
    ax = axes[4]
    ranks = np.arange(1, n_mus + 1)
    valid_first = case.first_spike_idx >= 0
    valid_last = case.last_spike_idx >= 0
    rec_val = np.full(n_mus, np.nan)
    derec_val = np.full(n_mus, np.nan)
    rec_time = np.full(n_mus, np.nan)
    derec_time = np.full(n_mus, np.nan)
    rec_val[valid_first] = case.ref_norm[case.first_spike_idx[valid_first]]
    derec_val[valid_last] = case.ref_norm[case.last_spike_idx[valid_last]]
    rec_time[valid_first] = case.first_spike_idx[valid_first] / fs
    derec_time[valid_last] = case.last_spike_idx[valid_last] / fs

    ax.scatter(ranks, rec_val, s=24, label="recruitment ref", zorder=3)
    ax.scatter(ranks, derec_val, s=24, marker="v", label="derecruitment ref", zorder=3)
    ax.plot(ranks, rec_val, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.plot(ranks, derec_val, linestyle="--", linewidth=0.6, alpha=0.45)
    ax.set_title("Reference value at first/last discharge, sorted by recruitment time")
    ax.set_xlabel("Recruitment rank")
    ax.set_ylabel("Normalized ref")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    # Add compact text summary to bottom panel
    summary = summarize_case(case)
    text = (
        f"duration={summary['duration_s']:.1f}s, spikes={summary['total_spikes']}, "
        f"mean FR all={summary['mean_fr_all_duration_hz_per_mu']:.2f}Hz/MU, "
        f"active FR={summary['mean_fr_active_duration_hz']:.2f}Hz"
    )
    ax.text(0.01, -0.35, text, transform=ax.transAxes, fontsize=9, va="top")

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_path = output_dir / f"{info.dataset}_exp_overview.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def plot_all_cases_overview(cases: Sequence[ExpCaseData], output_dir: Path, show: bool = False) -> Path:
    """Create a compact 2x2 overview of normalized reference and CST drive for all cases."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(cases)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.2 * nrows), squeeze=False)
    fig.suptitle("Experimental data overview: normalized reference and smoothed CST", fontsize=15, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    for ax, case in zip(axes.flat, cases):
        ax.axis("on")
        ax.plot(case.time, case.ref_norm, color="black", linewidth=1.2, label="reference")
        ax.plot(case.time, normalize_minmax(case.common_control_4hz), linewidth=1.1, label="CST 4 Hz")
        ax.axvspan(case.info.plateau_time1, case.info.plateau_time2, alpha=0.10)
        ax.set_title(f"{case.info.dataset} ({case.info.muscle}, {case.n_mus} MUs)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized")
        ax.set_xlim(case.time[0], case.time[-1])
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = output_dir / "overview_all_cases.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


def write_summary_csv(rows: Sequence[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "summary_all_cases.csv"
    if not rows:
        return out_path
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def print_summary(rows: Sequence[dict]) -> None:
    print("\nSummary")
    print("=" * 100)
    header = (
        f"{'dataset':10s} {'muscle':6s} {'MUs':>4s} {'duration(s)':>11s} "
        f"{'spikes':>8s} {'FR all':>8s} {'FR active':>10s} "
        f"{'first(s)':>8s} {'last(s)':>8s} {'rec ref':>8s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dataset']:10s} {r['muscle']:6s} {int(r['n_mus']):4d} "
            f"{float(r['duration_s']):11.2f} {int(r['total_spikes']):8d} "
            f"{float(r['mean_fr_all_duration_hz_per_mu']):8.2f} "
            f"{float(r['mean_fr_active_duration_hz']):10.2f} "
            f"{float(r['first_identified_spike_s']):8.2f} "
            f"{float(r['last_identified_spike_s']):8.2f} "
            f"{float(r['mean_recruitment_ref_norm']):8.2f}"
        )
    print("=" * 100)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all experimental MN/MU discharge .mat files in Input_Exp_Data/."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Input_Exp_Data"),
        help="Directory containing GM_30.mat, TA_35_D.mat, TA_35_H.mat, and TA_50.mat.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Exp_Data_Visualization"),
        help="Directory where figures and summary CSV will be saved.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(DEFAULT_CASES),
        help="Cases to plot. Default: all four cases.",
    )
    parser.add_argument(
        "--index-base",
        choices=["original", "zero", "one", "auto"],
        default="original",
        help=(
            "How to interpret MUPulses sample indices. "
            "'original' matches the original code and does no shift. "
            "'one' subtracts 1. 'auto' subtracts 1 only if needed to avoid out-of-bounds."
        ),
    )
    parser.add_argument(
        "--smoothing-sigma-ms",
        type=float,
        default=50.0,
        help="Gaussian smoothing sigma for CST visualization, in milliseconds.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    parser.add_argument(
        "--skip-overview",
        action="store_true",
        help="Skip the combined overview figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    sigma_s = args.smoothing_sigma_ms / 1000.0

    print(f"Input directory : {input_dir.resolve()}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Index base mode : {args.index_base}")

    loaded_cases: List[ExpCaseData] = []
    summaries: List[dict] = []

    for case_name in args.cases:
        print(f"\nLoading {case_name}...")
        try:
            case = load_case(input_dir, case_name, args.index_base, sigma_s)
        except Exception as exc:
            print(f"  WARNING: skipped {case_name}: {exc}")
            continue

        loaded_cases.append(case)
        row = summarize_case(case)
        summaries.append(row)
        fig_path = plot_case(case, output_dir, show=args.show)
        print(f"  saved {fig_path}")
        print(
            f"  {case.n_mus} MUs, duration={row['duration_s']:.2f}s, "
            f"spikes={row['total_spikes']}, "
            f"mean FR={row['mean_fr_all_duration_hz_per_mu']:.2f} Hz/MU over full trial"
        )

    if loaded_cases and not args.skip_overview:
        overview_path = plot_all_cases_overview(loaded_cases, output_dir, show=args.show)
        print(f"\nSaved overview: {overview_path}")

    if summaries:
        csv_path = write_summary_csv(summaries, output_dir)
        print(f"Saved summary CSV: {csv_path}")
        print_summary(summaries)
    else:
        print("No cases were loaded. Please check --input-dir and file names.")


if __name__ == "__main__":
    main()
