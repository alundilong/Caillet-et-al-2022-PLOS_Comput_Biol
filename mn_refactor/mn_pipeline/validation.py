from __future__ import annotations

import random
import time as walltime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import TestCase, _normalize_yes_no
from .current import CurrentInput
from .paths import repository_context
from .plotting import capture_matplotlib_show
from .workflow import StepTimer


VALIDATION_TEST_CASES: Dict[str, TestCase] = {
    # The validation script uses the same experimental trials as 1_MAIN but keeps
    # the original full-pool value of 550 for GM_30.
    "TA_35_D": TestCase("DelVecchio", "TA_35_D", "TA", 10.3, 20.5, 30.0, 0.35, 400, 2048),
    "TA_35_H": TestCase("Hug", "TA_35_H", "TA", 10.5, 20.5, 30.0, 0.35, 400, 2048),
    "TA_50": TestCase("Hug", "TA_50", "TA", 12.0, 21.8, 34.5, 0.50, 400, 2048),
    "GM_30": TestCase("Hug", "GM_30", "GM", 9.1, 19.1, 33.5, 0.30, 550, 2048),
}


@dataclass
class ValidationConfig:
    """Configuration for leave-one-MN-out validation.

    This corresponds to the original 2_MN_Model_validation.py workflow:
    hold out one experimental spike train, rebuild the model from the remaining
    Nr-1 trains, predict the held-out MN discharge, and compare simulated vs.
    experimental FIDF.
    """

    dataset: str = "TA_35_D"
    repository_root: Path = Path(".")
    mn_pop: Optional[int] = None
    cm_derec: float = 2.0e-2
    adapt_kR: str = "y"
    plot: str = "y"
    save: str = "y"
    seed: Optional[int] = None
    save_figures: bool = False
    figures_dir: Path = Path("figures_validation")
    close_saved_figures: bool = True
    plot_fold_details: bool = True

    def __post_init__(self) -> None:
        self.adapt_kR = _normalize_yes_no(self.adapt_kR, yes="y", no="n")
        self.plot = _normalize_yes_no(self.plot, yes="y", no="n")
        self.save = _normalize_yes_no(self.save, yes="y", no="n")
        if self.dataset not in VALIDATION_TEST_CASES:
            available = ", ".join(sorted(VALIDATION_TEST_CASES))
            raise ValueError(f"Unknown dataset {self.dataset!r}. Available datasets: {available}")

    @property
    def test_case(self) -> TestCase:
        return VALIDATION_TEST_CASES[self.dataset]

    @property
    def effective_mn_pop(self) -> int:
        return int(self.mn_pop if self.mn_pop is not None else self.test_case.mn_pop)


@dataclass
class ValidationResults:
    dataset: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, Any] = field(default_factory=dict)
    runtime_minutes: Optional[float] = None


def _as_single_object_array(spikes: np.ndarray) -> np.ndarray:
    """Wrap a single spike-train vector so modules see one MN, not one scalar."""
    arr = np.empty((1,), dtype=object)
    arr[0] = np.asarray(spikes)
    return arr


def _safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(a.size, b.size)
    if n < 2:
        return np.nan
    a = a[:n]
    b = b[:n]
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def _safe_normalized_rmse(rms_func, target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    n = min(target.size, pred.size)
    if n == 0 or np.nanmax(target[:n]) <= 0:
        return np.nan
    return float(rms_func(target[:n], pred[:n]) / np.nanmax(target[:n]) * 100.0)


def _safe_max_normalized_error(target: np.ndarray, pred: np.ndarray, eps: float = 1e-4) -> float:
    target = np.asarray(target, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    n = min(target.size, pred.size)
    if n == 0:
        return np.nan
    target = target[:n]
    pred = pred[:n]
    non_zero = np.flatnonzero(target > eps)
    if non_zero.size == 0:
        return np.nan
    return float(np.nanmax(np.abs((pred[non_zero] - target[non_zero]) / target[non_zero] * 100.0)))


class LeaveOneOutValidationPipeline:
    """Readable refactor of 2_MN_Model_validation.py.

    The class preserves the original leave-one-out scientific workflow while
    avoiding repeated force preprocessing and removing plot side effects from
    current-input construction.
    """

    def __init__(self, config: ValidationConfig):
        self.cfg = config
        self.case = config.test_case
        self.res = ValidationResults(dataset=config.dataset)

    def run(self) -> ValidationResults:
        t0 = walltime.time()
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)
            random.seed(self.cfg.seed)
            print(f"Random seed set to {self.cfg.seed}")

        with repository_context(self.cfg.repository_root):
            with capture_matplotlib_show(
                enabled=self.cfg.save_figures,
                figures_dir=self.cfg.repository_root / self.cfg.figures_dir,
                close=self.cfg.close_saved_figures,
            ):
                self._run_validation()

        self.res.runtime_minutes = (walltime.time() - t0) / 60.0
        print(f"\nMy validation program took {self.res.runtime_minutes:.3f} minutes to run")
        return self.res

    # ------------------------------------------------------------------
    # Main validation workflow
    # ------------------------------------------------------------------
    def _run_validation(self) -> None:
        self._load_and_preprocess_once()
        self._allocate_outputs()

        for held_out_idx in range(self.n_total_mn):
            self._run_one_fold(held_out_idx)

        self._plot_final_summary_if_requested()
        self._save_outputs_if_requested()

    def _load_and_preprocess_once(self) -> None:
        from EXP_DATA_PROCESSING_MOD import EXP_DATA_PROCESSING_func
        from Reshaping_MOD import preprocessing_func

        with StepTimer("1. Loading and preprocessing complete experimental dataset"):
            self.n_total_mn, force_raw, self.disch_times_complete = EXP_DATA_PROCESSING_func(
                self.case.author,
                self.case.name,
            )
            (
                self.Force,
                self.time,
                self.MN_list,
                self.t_start,
                self.t_stop,
                self.t_stop_calib,
                self.kR,
                self.Cm_rec,
                self.step_size,
            ) = preprocessing_func(
                self.case.author,
                force_raw,
                self.case.end_force,
                self.case.fs,
                self.n_total_mn,
                self.case.plateau_time1,
                self.case.plateau_time2,
            )
            # The original validation script calibrates only over the first half
            # of the plateau.
            self.t_stop_calib = (self.case.plateau_time1 + self.case.plateau_time2) / 2.0
            self.fs = int(self.case.fs)
            self.MN_pop = self.cfg.effective_mn_pop
            self.MN_pool_list = np.arange(0, self.MN_pop, 1)
            self.range_start = int(self.t_start * self.fs)
            self.range_stop = int(self.t_stop * self.fs)

            print(f"Dataset: {self.case.name}")
            print(f"Author/source: {self.case.author}")
            print(f"Muscle: {self.case.muscle}")
            print(f"Recorded MNs: {self.n_total_mn}")
            print(f"Full MN pool used for mapping: {self.MN_pop}")
            print(f"Cm_derec: {self.cfg.cm_derec}")
            print(f"kR adapted for derecruitment? {self.cfg.adapt_kR}")

            self.res.arrays.update(
                Force=self.Force,
                time=self.time,
                disch_times_complete=self.disch_times_complete,
            )

    def _allocate_outputs(self) -> None:
        n = self.n_total_mn
        self.deltaf1_table_sim = np.empty((n,), dtype=object)
        self.RMS_table_sim = np.empty((n,), dtype=object)
        self.nME_table_sim = np.empty((n,), dtype=object)
        self.Corrcoef_table_sim = np.empty((n,), dtype=object)
        self.Size_distrib_matrix = np.zeros((n, 2), dtype=float)
        self.FIDF_sim_test_array = np.empty((n,), dtype=object)
        self.FIDF_exp_test_array = np.empty((n,), dtype=object)
        self.Real_MN_pop_test_array = np.empty((n,), dtype=object)
        self.Firing_times_sim_test_array = np.empty((n,), dtype=object)

    def _run_one_fold(self, held_out_idx: int) -> None:
        fold_t0 = walltime.time()
        print("\n# --------------------------------------------------------------------")
        print(f"Leave-one-out fold {held_out_idx + 1}/{self.n_total_mn}: hold out identified MN #{held_out_idx + 1}")

        train_disch_times = np.delete(self.disch_times_complete, held_out_idx, axis=0)
        test_disch_times = self.disch_times_complete[held_out_idx]
        test_disch_obj = _as_single_object_array(test_disch_times)
        n_train = self.n_total_mn - 1

        # 1. CST and common inputs from training MUs only.
        from CST_MOD import CST_func
        from But_filter_MOD import But_filter_func
        from EXP_THRESHOLDS_MOD import exp_thresholds_func
        from MN_distirbution_MOD import MN_distirbution_func
        from IDF_MOD import IDF_func
        from Hanning_filter_MOD import Hanning_filter_func
        from exp_ARP_MOD import exp_ARP_func
        from ARP_distrib_MOD import ARP_distrib_func
        from Curr_input_MOD import Common_to_current_input_func
        from MN_properties_relationships_MOD import S_Ith
        from Simplified_Size_calibration_MOD import Size_calibration_function
        from calibrated_sizes_MOD import calibrated_sizes_func
        from Virtual_pop_size_arp_MOD import Virtual_size_arp_pop
        from RC_LIF_MOD import RC_solve_func
        from RMS_func import RMS_func
        import PLOTS as plots

        Binary_matrix_exp, CST_exp = CST_func(n_train, self.time, train_disch_times, fs=self.fs)
        common_input_exp = But_filter_func(10, CST_exp, fs=self.fs)
        common_control_exp = But_filter_func(4, CST_exp, fs=self.fs)

        # 2. Thresholds and real-pool locations for training and held-out MNs.
        THRESHOLDS = exp_thresholds_func(
            n_train,
            self.case.mvc,
            train_disch_times,
            common_input_exp,
            self.Force,
            fs=self.fs,
        )
        THRESHOLDS_test = exp_thresholds_func(
            1,
            self.case.mvc,
            test_disch_obj,
            common_input_exp,
            self.Force,
            fs=self.fs,
        )
        popt_th = plots.plot_force_thresholds_func(
            THRESHOLDS[:, 2],
            THRESHOLDS[:, 5],
            self.case.muscle,
            self.case.mvc,
            "n",
            self.MN_pop,
        )
        kR_derec = self.kR / popt_th[0]
        Real_MN_pop = MN_distirbution_func(n_train, self.MN_pop, self.MN_pool_list, self.case.muscle, THRESHOLDS)
        Real_MN_pop_test = MN_distirbution_func(1, self.MN_pop, self.MN_pool_list, self.case.muscle, THRESHOLDS_test)
        test_pool_idx = int(np.asarray(Real_MN_pop_test).ravel()[0])

        # 3. Experimental FIDF for training and held-out MNs.
        IDF_dt, FF_FULL = IDF_func(n_train, self.time, train_disch_times, self.t_start, self.t_stop, fs=self.fs)
        FIDF_exp = Hanning_filter_func(n_train, FF_FULL, fs=self.fs)
        IDF_dt_exp_test, FF_FULL_exp_test = IDF_func(1, self.time, test_disch_times, self.t_start, self.t_stop, fs=self.fs)
        FIDF_exp_test = Hanning_filter_func(1, FF_FULL_exp_test, fs=self.fs)

        # 4. Training ARP distribution.
        saturating_MN, exp_ARP = exp_ARP_func(
            n_train,
            self.case.plateau_time1,
            self.case.plateau_time2,
            self.case.end_force,
            train_disch_times,
            IDF_dt,
            fs=self.fs,
        )
        a_arp, b_arp, ARP_table, saturating_MN, Non_saturating_MN = ARP_distrib_func(
            self.case.name,
            n_train,
            self.MN_pop,
            Real_MN_pop,
            saturating_MN,
            exp_ARP,
            self.case.muscle,
            "n",
            self.MN_pop,
        )
        if self.cfg.plot == "y" and self.cfg.plot_fold_details:
            plots.plot_MN_saturation_func(saturating_MN, Non_saturating_MN, Real_MN_pop, ARP_table)

        # 5. Current input. Use fast array-backed current instead of plotting to
        # obtain I_smooth_list.
        G, I1 = Common_to_current_input_func(THRESHOLDS, Real_MN_pop, self.MN_pop, common_input_exp, fs=self.fs)
        I = CurrentInput(common_input_exp, THRESHOLDS, G, I1, fs=self.fs)
        I_smooth = CurrentInput(common_control_exp, THRESHOLDS, G, I1, fs=self.fs)
        I_smooth_array = I_smooth.as_array()
        if self.cfg.plot == "y" and self.cfg.plot_fold_details:
            plots.plot_current_input(self.time, I_smooth, self.case.end_force)

        # 6. Calibrate sizes from the training set.
        plateau_slice = slice(int(self.case.plateau_time1 * self.fs), int(self.case.plateau_time2 * self.fs))
        Size_min = 0.08e-6
        Size_max = S_Ith(np.min(I_smooth_array[plateau_slice]))

        print("Performing size calibration from training MUs...")
        sol = Size_calibration_function(
            self.Cm_rec,
            self.time,
            self.t_start,
            self.t_stop_calib,
            self.case.plateau_time2,
            self.case.end_force,
            Size_min,
            Size_max,
            self.step_size,
            self.kR,
            n_train,
            FIDF_exp,
            ARP_table,
            I,
            plot="n",
            fs=self.fs,
        )
        Calib_sizes = sol[0]

        # 7. Fit size distribution over the full pool and predict the held-out MN.
        Real_MN_pop, Calib_sizes, popt, r2_size_distribution = calibrated_sizes_func(
            Real_MN_pop,
            Calib_sizes,
            self.case.muscle,
            self.MN_pop,
        )
        a_size = round(popt[0], 11)
        c_size = round(popt[1], 4)

        def size_power(x, a, c):
            return a * 2.4 ** (((x + 1) / self.MN_pop) ** c)

        if self.cfg.plot == "y" and self.cfg.plot_fold_details:
            plots.plot_size_distribution_func(
                Real_MN_pop,
                size_power,
                popt,
                a_size,
                c_size,
                r2_size_distribution,
                Calib_sizes,
                self.MN_pop,
            )

        Virtual_size_arr, Virtual_ARP_arr = Virtual_size_arp_pop(self.MN_pop, a_size, c_size, exp_ARP, a_arp, b_arp)
        test_size = float(np.asarray(Virtual_size_arr[test_pool_idx]).ravel()[0])
        test_arp = float(np.asarray(Virtual_ARP_arr[test_pool_idx]).ravel()[0])

        print(f"Simulating held-out MN #{held_out_idx + 1} mapped to full-pool index {test_pool_idx}...")
        _, _, firing_times_sim_test, _ = RC_solve_func(
            I,
            self.t_start,
            self.t_stop,
            self.case.plateau_time2,
            test_size,
            self.Cm_rec,
            self.cfg.cm_derec,
            self.step_size,
            test_arp,
            self.kR,
            self.cfg.adapt_kR,
            kR_derec,
        )
        _, FF_FULL_sim_test = IDF_func(1, self.time, firing_times_sim_test, self.t_start, self.t_stop, fs=self.fs)
        FIDF_sim_test = Hanning_filter_func(1, FF_FULL_sim_test, fs=self.fs)

        # 8. Metrics for held-out MN.
        if firing_times_sim_test.size > 0:
            delta_tf1_sim_test = float(firing_times_sim_test[0] - THRESHOLDS_test[0, 0])
        else:
            delta_tf1_sim_test = np.nan

        nME = _safe_max_normalized_error(FIDF_exp_test, FIDF_sim_test)
        RMS_normalized = _safe_normalized_rmse(RMS_func, FIDF_exp_test, FIDF_sim_test)
        r2_validation = _safe_r2(FIDF_sim_test, FIDF_exp_test)

        self.deltaf1_table_sim[held_out_idx] = delta_tf1_sim_test
        self.nME_table_sim[held_out_idx] = nME
        self.RMS_table_sim[held_out_idx] = RMS_normalized
        self.Corrcoef_table_sim[held_out_idx] = r2_validation
        self.Size_distrib_matrix[held_out_idx] = np.array([a_size, c_size])
        self.FIDF_sim_test_array[held_out_idx] = FIDF_sim_test
        self.FIDF_exp_test_array[held_out_idx] = FIDF_exp_test
        self.Real_MN_pop_test_array[held_out_idx] = test_pool_idx
        self.Firing_times_sim_test_array[held_out_idx] = firing_times_sim_test

        if self.cfg.plot == "y":
            plots.plot_exp_vs_sim_FIDF_func(
                self.time,
                self.range_start,
                FIDF_exp_test,
                FIDF_sim_test,
                held_out_idx,
                "validation",
            )

        print(
            f"Held-out MN #{held_out_idx + 1} mapped to pool index {test_pool_idx}: "
            f"Deltaf1={delta_tf1_sim_test:.3g} s, "
            f"nRMSE={RMS_normalized:.3g}%, "
            f"r2={r2_validation:.3g}, "
            f"fold runtime={walltime.time() - fold_t0:.2f} s"
        )

    def _plot_final_summary_if_requested(self) -> None:
        if self.cfg.plot != "y":
            return
        import PLOTS as plots

        with StepTimer("7. Plotting validation summary"):
            plots.plot_final_results_validation_func(
                self.n_total_mn - 1,
                self.deltaf1_table_sim,
                self.nME_table_sim,
                self.RMS_table_sim,
                self.Corrcoef_table_sim,
                self.Size_distrib_matrix,
            )

    def _save_outputs_if_requested(self) -> None:
        if self.cfg.save != "y":
            return

        with StepTimer("8. Saving validation outputs"):
            prefix = f"{self.case.author}_{self.case.name}_validation_"
            np.save(prefix + "onset_error", self.deltaf1_table_sim, allow_pickle=True)
            np.save(prefix + "nRMSE", self.RMS_table_sim, allow_pickle=True)
            np.save(prefix + "r2", self.Corrcoef_table_sim, allow_pickle=True)
            np.save(prefix + "Size_distributions", self.Size_distrib_matrix, allow_pickle=True)
            np.save(prefix + "FIDF_sim_test", self.FIDF_sim_test_array, allow_pickle=True)
            np.save(prefix + "FIDF_exp_test", self.FIDF_exp_test_array, allow_pickle=True)
            np.save(prefix + "Real_MN_pop_test", self.Real_MN_pop_test_array, allow_pickle=True)
            np.save(prefix + "firing_times_sim_test", self.Firing_times_sim_test_array, allow_pickle=True)

            self.res.arrays.update(
                deltaf1_table_sim=self.deltaf1_table_sim,
                RMS_table_sim=self.RMS_table_sim,
                nME_table_sim=self.nME_table_sim,
                Corrcoef_table_sim=self.Corrcoef_table_sim,
                Size_distrib_matrix=self.Size_distrib_matrix,
                FIDF_sim_test_array=self.FIDF_sim_test_array,
                FIDF_exp_test_array=self.FIDF_exp_test_array,
                Real_MN_pop_test_array=self.Real_MN_pop_test_array,
                Firing_times_sim_test_array=self.Firing_times_sim_test_array,
            )

