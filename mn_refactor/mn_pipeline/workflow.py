from __future__ import annotations

import random
import time as walltime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .config import RunConfig
from .current import CurrentInput
from .paths import repository_context
from .plotting import capture_matplotlib_show


@dataclass
class PipelineResults:
    """Container for the most important pipeline outputs."""

    dataset: str
    metrics: Dict[str, float] = field(default_factory=dict)
    arrays: Dict[str, Any] = field(default_factory=dict)
    runtime_minutes: Optional[float] = None


class StepTimer:
    def __init__(self, title: str):
        self.title = title
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = walltime.time()
        print(f"\n# --------------------------------------------------------------------")
        print(self.title)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = walltime.time() - self.t0
        print(f"Finished: {self.title} ({dt:.2f} s)")
        return False


class MotoneuronReconstructionPipeline:
    """
    Refactored workflow corresponding to the original 1_MAIN_MN_model.py.

    The scientific sequence is preserved, but the implementation is split into
    named steps and uses a faster callable current input plus vectorized helper
    modules where those changes do not alter the algorithm.
    """

    def __init__(self, config: RunConfig):
        self.cfg = config
        self.case = config.test_case
        self.res = PipelineResults(dataset=config.dataset)

    def run(self) -> PipelineResults:
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
                self._run_steps()

        self.res.runtime_minutes = (walltime.time() - t0) / 60.0
        print(f"\nMy program took {self.res.runtime_minutes:.3f} minutes to run")
        return self.res

    # ------------------------------------------------------------------
    # Main workflow
    # ------------------------------------------------------------------
    def _run_steps(self) -> None:
        self._load_and_preprocess()
        self._compute_experimental_cst_and_common_inputs()
        self._coherence_thresholds_and_pool_locations()
        self._compute_experimental_fidfs_and_arp()
        self._build_current_input()
        self._calibrate_sizes()
        self._optional_cm_derec_sensitivity()
        self._fit_complete_size_distribution()
        self._simulate_complete_pool()
        self._global_validation()
        self._save_outputs_if_requested()

    def _load_and_preprocess(self) -> None:
        from EXP_DATA_PROCESSING_MOD import EXP_DATA_PROCESSING_func
        from Reshaping_MOD import preprocessing_func

        with StepTimer("1.1 Loading and preprocessing experimental data"):
            self.Nb_MN, force_raw, self.disch_times = EXP_DATA_PROCESSING_func(self.case.author, self.case.name)
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
                self.Nb_MN,
                self.case.plateau_time1,
                self.case.plateau_time2,
            )

            self.MN_pop = int(self.cfg.mn_pop)
            self.true_MN_pop = int(self.cfg.true_mn_pop or self.cfg.mn_pop)
            self.MN_pool_list = np.arange(0, self.true_MN_pop, 1)
            self.fs = int(self.case.fs)
            self.Cm_derec = float(self.cfg.cm_derec)

            print(f"Dataset: {self.case.name}")
            print(f"Author/source: {self.case.author}")
            print(f"Muscle: {self.case.muscle}")
            print(f"Recorded MNs: {self.Nb_MN}")
            print(f"Simulated MN pool size: {self.MN_pop}")
            print(f"True MN pool used for mapping: {self.true_MN_pop}")
            print(f"Cm_derec calibrated? {self.cfg.cm_derec_calib}")
            print(f"kR adapted for derecruitment? {self.cfg.adapt_kR}")

            self.res.arrays.update(
                Force=self.Force,
                time=self.time,
                disch_times=self.disch_times,
                MN_list=self.MN_list,
            )

    def _compute_experimental_cst_and_common_inputs(self) -> None:
        from CST_MOD import CST_func
        from But_filter_MOD import But_filter_func
        import PLOTS as plots

        with StepTimer("1.2 Computing experimental CST, common input, and common control"):
            self.Binary_matrix_exp, self.CST_exp = CST_func(self.Nb_MN, self.time, self.disch_times, fs=self.fs)
            self.common_input_exp = But_filter_func(10, self.CST_exp, fs=self.fs)
            self.common_control_exp = But_filter_func(4, self.CST_exp, fs=self.fs)
            self.common_noise_exp = self.common_input_exp - self.common_control_exp

            if self.cfg.plot == "y":
                plots.plot_CST_func(self.time, self.CST_exp, self.case.end_force)
                plots.plot_common_inputs_func(
                    self.time,
                    self.common_input_exp,
                    self.common_control_exp,
                    self.common_noise_exp,
                    self.case.end_force,
                )

            self.res.arrays.update(
                Binary_matrix_exp=self.Binary_matrix_exp,
                CST_exp=self.CST_exp,
                common_input_exp=self.common_input_exp,
                common_control_exp=self.common_control_exp,
                common_noise_exp=self.common_noise_exp,
            )

    def _coherence_thresholds_and_pool_locations(self) -> None:
        from subset_coher_MOD import subset_coher_func
        from EXP_THRESHOLDS_MOD import exp_thresholds_func
        from MN_distirbution_MOD import MN_distirbution_func
        import PLOTS as plots

        with StepTimer("1.3 Coherence, thresholds, and MN locations in the real pool"):
            print("Assessing coherence between experimental CSTs")
            self.avg_all_coher = subset_coher_func(
                self.cfg.nb_coherence_tests,
                self.time,
                self.Nb_MN,
                self.Binary_matrix_exp,
                fs=self.fs,
            )
            print(f"Average coherence after {self.cfg.nb_coherence_tests} subset trials: {self.avg_all_coher}")

            self.THRESHOLDS = exp_thresholds_func(
                self.Nb_MN,
                self.case.mvc,
                self.disch_times,
                self.common_input_exp,
                self.Force,
                fs=self.fs,
            )
            self.popt_th = plots.plot_force_thresholds_func(
                self.THRESHOLDS[:, 2],
                self.THRESHOLDS[:, 5],
                self.case.muscle,
                self.case.mvc,
                self.cfg.plot,
                self.MN_pop,
            )
            self.kR_derec = self.kR / self.popt_th[0]

            self.Real_MN_pop = MN_distirbution_func(
                self.Nb_MN,
                self.true_MN_pop,
                self.MN_pool_list,
                self.case.muscle,
                self.THRESHOLDS,
            )
            if self.cfg.plot == "y":
                plots.plot_loc_in_real_MN_pop(self.MN_list, self.Real_MN_pop, self.true_MN_pop)

            self.res.metrics["avg_subset_coherence"] = float(self.avg_all_coher)
            self.res.arrays.update(THRESHOLDS=self.THRESHOLDS, Real_MN_pop=self.Real_MN_pop)

    def _compute_experimental_fidfs_and_arp(self) -> None:
        from IDF_MOD import IDF_func
        from Hanning_filter_MOD import Hanning_filter_func
        from exp_ARP_MOD import exp_ARP_func
        from ARP_distrib_MOD import ARP_distrib_func
        import PLOTS as plots

        with StepTimer("1.4-1.5 Experimental FIDFs and inert-period/ARP distribution"):
            self.IDF_dt, self.FF_FULL = IDF_func(
                self.Nb_MN,
                self.time,
                self.disch_times,
                self.t_start,
                self.t_stop,
                fs=self.fs,
            )
            self.FIDF_exp = Hanning_filter_func(self.Nb_MN, self.FF_FULL, fs=self.fs)

            if self.cfg.plot == "y" and self.Nb_MN > 0:
                k = 0
                plots.plot_IDF_FIDF_func(
                    self.time,
                    self.disch_times[k],
                    self.IDF_dt[k],
                    self.FF_FULL[k],
                    self.FIDF_exp[k],
                    self.case.end_force,
                    fs=self.fs,
                )

            self.saturating_MN, self.exp_ARP = exp_ARP_func(
                self.Nb_MN,
                self.case.plateau_time1,
                self.case.plateau_time2,
                self.case.end_force,
                self.disch_times,
                self.IDF_dt,
                fs=self.fs,
            )
            (
                self.a_arp,
                self.b_arp,
                self.ARP_table,
                self.saturating_MN,
                self.Non_saturating_MN,
            ) = ARP_distrib_func(
                self.case.name,
                self.Nb_MN,
                self.true_MN_pop,
                self.Real_MN_pop,
                self.saturating_MN,
                self.exp_ARP,
                self.case.muscle,
                self.cfg.plot,
                self.MN_pop,
            )
            if self.cfg.plot == "y":
                try:
                    plots.plot_MN_saturation_func(
                        self.saturating_MN,
                        self.Non_saturating_MN,
                        self.Real_MN_pop,
                        self.ARP_table,
                    )
                except Exception as exc:
                    print(f"Skipping saturation plot: {exc}")

                try:
                    plots.plot_spike_trains_force_CI_func(
                        self.Nb_MN,
                        self.Force,
                        self.time,
                        self.common_input_exp,
                        self.common_control_exp,
                        self.t_start,
                        self.case.end_force,
                        self.fs,
                        self.disch_times,
                        self.case.mvc,
                    )
                except Exception as exc:
                    print(f"Skipping experimental spike-train/force/CI plot: {exc}")

            self.res.arrays.update(
                IDF_dt=self.IDF_dt,
                FF_FULL=self.FF_FULL,
                FIDF_exp=self.FIDF_exp,
                ARP_table=self.ARP_table,
                exp_ARP=self.exp_ARP,
            )

    def _build_current_input(self) -> None:
        from Curr_input_MOD import Common_to_current_input_func
        import PLOTS as plots

        with StepTimer("2. Computing common synaptic current input I(t)"):
            self.G, self.I1 = Common_to_current_input_func(
                self.THRESHOLDS,
                self.Real_MN_pop,
                self.true_MN_pop,
                self.common_input_exp,
                fs=self.fs,
            )
            self.I = CurrentInput(self.common_input_exp, self.THRESHOLDS, self.G, self.I1, fs=self.fs)
            self.I_smooth = CurrentInput(self.common_control_exp, self.THRESHOLDS, self.G, self.I1, fs=self.fs)

            self.I_array = self.I.as_array()
            self.I_smooth_array = self.I_smooth.as_array()
            self.I_list_nA = self.I.as_nA()
            self.I_smooth_list_nA = self.I_smooth.as_nA()

            if self.cfg.plot == "y":
                # Keep original plotting behavior. We do not depend on its return
                # values because those are recomputed above for plot='n' support.
                plots.plot_current_input(self.time, self.I, self.case.end_force)
                plots.plot_current_input(self.time, self.I_smooth, self.case.end_force)

            self.res.arrays.update(I_array=self.I_array, I_smooth_array=self.I_smooth_array)

    def _calibrate_sizes(self) -> None:
        from MN_properties_relationships_MOD import S_Ith
        from Simplified_Size_calibration_MOD import Size_calibration_function
        from delta_ft1_noncalib_MOD import delta_ft1_noncalib_func
        import PLOTS as plots

        with StepTimer("3. Calibrating identified MN sizes"):
            range_plateau_start = int(self.case.plateau_time1 * self.fs)
            range_plateau_stop = int(self.case.plateau_time2 * self.fs)
            min_i_smooth_plateau = np.min(self.I_smooth_array[range_plateau_start:range_plateau_stop])

            self.Size_min = 0.08e-6
            self.Size_max = S_Ith(min_i_smooth_plateau)
            self.Size_start = S_Ith(10e-9)

            print(f"Size bounds: [{self.Size_min:.6e}, {self.Size_max:.6e}] m^2")
            self.sol = Size_calibration_function(
                self.Cm_rec,
                self.time,
                self.t_start,
                self.t_stop_calib,
                self.case.plateau_time2,
                self.case.end_force,
                self.Size_min,
                self.Size_max,
                self.step_size,
                self.kR,
                self.Nb_MN,
                self.FIDF_exp,
                self.ARP_table,
                self.I,
                plot=self.cfg.plot,
                fs=self.fs,
            )
            self.Calib_sizes = self.sol[0]
            self.Calib_RMS_table = self.sol[2]
            self.Calib_r2_table = self.sol[3]

            self.Calib_delta_tf1 = delta_ft1_noncalib_func(
                self.Nb_MN,
                self.I,
                self.t_start,
                self.t_stop,
                self.case.plateau_time2,
                self.Calib_sizes,
                self.Cm_rec,
                self.Cm_rec,
                self.step_size,
                self.ARP_table,
                self.THRESHOLDS,
            )
            if self.cfg.plot == "y":
                plots.plot_achieved_error_S_calibration(
                    self.Nb_MN,
                    self.Calib_sizes,
                    self.Calib_delta_tf1,
                    self.Calib_RMS_table,
                    self.Calib_r2_table,
                )

            self.res.arrays.update(
                Calib_sizes=self.Calib_sizes,
                Calib_RMS_table=self.Calib_RMS_table,
                Calib_r2_table=self.Calib_r2_table,
                Calib_delta_tf1=self.Calib_delta_tf1,
            )

    def _optional_cm_derec_sensitivity(self) -> None:
        import PLOTS as plots

        with StepTimer("3a. Optional Cm_derec sensitivity analysis"):
            if self.cfg.cm_derec_calib == "yes":
                from Cm_derec_sensitivity_MOD import Cm_derec_sensitivity_func

                (
                    self.Cm_derec_array,
                    self.RMS_normalized_table_derec,
                    self.r2_mean_table_derec,
                    self.Cm_derec,
                ) = Cm_derec_sensitivity_func(
                    self.Nb_MN,
                    self.time,
                    self.I,
                    self.Cm_rec,
                    self.cfg.cm_derec_array,
                    self.Calib_sizes,
                    self.ARP_table,
                    self.FIDF_exp,
                    self.case.plateau_time1 + (self.case.plateau_time2 - self.case.plateau_time1) / 2,
                    self.case.end_force,
                    self.step_size,
                    self.kR,
                    self.cfg.adapt_kR,
                    self.kR_derec,
                    fs=self.fs,
                )
                print(f"Best Cm_derec = {self.Cm_derec * 100}")
                if self.cfg.plot == "y":
                    plots.plot_Cm_derec_sensitivity_analysis_func(
                        self.Cm_derec_array,
                        self.r2_mean_table_derec,
                        self.RMS_normalized_table_derec,
                    )
            else:
                print(f"Cm_derec set to {self.Cm_derec * 100}")

            self.Calib_FIDF_list = np.empty((self.Nb_MN,), dtype=object)
            if self.cfg.plot == "y":
                for i in range(self.Nb_MN):
                    self.Calib_FIDF_list[i] = plots.plot_calibrated_FIDF(
                        i,
                        self.time,
                        self.FIDF_exp,
                        self.I,
                        self.t_start,
                        self.t_stop,
                        self.case.plateau_time2,
                        self.Calib_sizes,
                        self.Cm_rec,
                        self.Cm_derec,
                        self.step_size,
                        self.ARP_table,
                        self.kR,
                        self.cfg.adapt_kR,
                        self.kR_derec,
                        fs=self.fs,
                    )

    def _fit_complete_size_distribution(self) -> None:
        from calibrated_sizes_MOD import calibrated_sizes_func
        import PLOTS as plots

        with StepTimer("4. Fitting calibrated sizes back to complete MN population"):
            (
                self.Real_MN_pop,
                self.Calib_sizes,
                self.popt,
                self.r2_size_calib,
            ) = calibrated_sizes_func(
                self.Real_MN_pop,
                self.Calib_sizes.astype(float),
                self.case.muscle,
                self.MN_pop,
            )
            self.a_size = round(self.popt[0], 11)
            self.c_size = round(self.popt[1], 4)

            def size_power(x, a, c):
                return a * 2.4 ** (((x + 1) / self.true_MN_pop) ** c)

            self.size_power = size_power
            if self.cfg.plot == "y":
                plots.plot_size_distribution_func(
                    self.Real_MN_pop,
                    size_power,
                    self.popt,
                    self.a_size,
                    self.c_size,
                    self.r2_size_calib,
                    self.Calib_sizes.astype(float),
                    self.MN_pop,
                )

            self.res.metrics["r2_size_calib"] = float(self.r2_size_calib)
            self.res.arrays.update(Calib_sizes=self.Calib_sizes, Real_MN_pop=self.Real_MN_pop)

    def _simulate_complete_pool(self) -> None:
        from Virtual_pop_size_arp_MOD import Virtual_size_arp_pop
        from run_LIF_simulation_MOD import run_LIF_simulation_func
        from delta_ft1_calib_MOD import delta_ft1_calib_func

        with StepTimer("5. Simulating firing activity of the complete MN population"):
            self.range_start = int(self.t_start * self.fs)
            self.range_stop = int(self.t_stop * self.fs)
            self.Virtual_size_arr, self.Virtual_ARP_arr = Virtual_size_arp_pop(
                self.MN_pop,
                self.a_size,
                self.c_size,
                self.exp_ARP,
                self.a_arp,
                self.b_arp,
            )
            self.Sol_simulation = run_LIF_simulation_func(
                self.Nb_MN,
                self.MN_pop,
                self.Real_MN_pop,
                self.time,
                self.t_start,
                self.t_stop,
                self.case.plateau_time2,
                self.FF_FULL,
                self.FIDF_exp,
                self.Virtual_size_arr,
                self.Virtual_ARP_arr,
                self.I,
                self.Cm_rec,
                self.Cm_derec,
                self.step_size,
                self.kR,
                self.cfg.plot,
                self.cfg.adapt_kR,
                self.kR_derec,
            )
            self.nME_sim = self.Sol_simulation[0]
            self.RMS_table_sim = self.Sol_simulation[1]
            self.Corrcoef_table_sim = self.Sol_simulation[2]
            self.Firing_times_sim = self.Sol_simulation[3]
            self.delta_tf1_end = delta_ft1_calib_func(
                self.Nb_MN,
                self.Real_MN_pop,
                self.Firing_times_sim,
                self.THRESHOLDS,
            )
            self.res.arrays.update(
                Virtual_size_arr=self.Virtual_size_arr,
                Virtual_ARP_arr=self.Virtual_ARP_arr,
                Firing_times_sim=self.Firing_times_sim,
                delta_tf1_end=self.delta_tf1_end,
            )

    def _global_validation(self) -> None:
        from CST_MOD import CST_func
        from But_filter_MOD import But_filter_func
        from RMS_func import RMS_func
        import PLOTS as plots

        with StepTimer("6. Global validation: simulated neural drive vs force"):
            self.Binary_matrix_sim, self.CST_sim = CST_func(
                self.MN_pop,
                self.time[self.range_start : self.range_stop],
                self.Firing_times_sim,
                "sec",
                fs=self.fs,
            )
            self.common_control_sim = But_filter_func(4, self.CST_sim, fs=self.fs)
            self.common_input_sim = But_filter_func(10, self.CST_sim, fs=self.fs)

            if self.cfg.plot == "y":
                plots.plot_spike_trains_force_CI_func(
                    self.MN_pop,
                    self.Force,
                    self.time,
                    self.common_input_sim,
                    self.common_control_sim,
                    self.t_start,
                    self.case.end_force,
                    self.fs,
                    self.Firing_times_sim,
                    self.case.mvc,
                    "sim",
                )
                plots.plot_final_results_func(
                    self.Nb_MN,
                    self.time,
                    self.t_start,
                    self.range_start,
                    self.case.end_force,
                    self.range_stop,
                    self.common_control_exp,
                    self.common_control_sim,
                    self.common_input_exp,
                    self.common_input_sim,
                    self.Force,
                )

            exp_slice = slice(self.range_start, self.range_stop)
            self.r2_exp = round(np.corrcoef(self.common_control_exp[exp_slice], self.Force[exp_slice])[0][1] ** 2, 2)
            self.nRMSE_exp = (
                RMS_func(
                    self.common_control_exp[exp_slice] / max(self.common_control_exp[exp_slice]),
                    self.Force[exp_slice] / max(self.Force[exp_slice]),
                )
                * 100
            )
            self.r2_sim = round(np.corrcoef(self.common_control_sim, self.Force[exp_slice])[0][1] ** 2, 2)
            self.nRMSE_sim = (
                RMS_func(
                    self.common_control_sim / max(self.common_control_sim),
                    self.Force[exp_slice] / max(self.Force[exp_slice]),
                )
                * 100
            )

            print(f"Coef of determination between EXP force trace and common control = {self.r2_exp}")
            print(f"nRMSE EXP = {self.nRMSE_exp}")
            print(f"Coef of determination between SIM force trace and common control = {self.r2_sim}")
            print(f"nRMSE SIM = {self.nRMSE_sim}")

            self.res.metrics.update(
                r2_exp=float(self.r2_exp),
                nRMSE_exp=float(self.nRMSE_exp),
                r2_sim=float(self.r2_sim),
                nRMSE_sim=float(self.nRMSE_sim),
            )
            self.res.arrays.update(
                Binary_matrix_sim=self.Binary_matrix_sim,
                CST_sim=self.CST_sim,
                common_control_sim=self.common_control_sim,
                common_input_sim=self.common_input_sim,
            )

    def _save_outputs_if_requested(self) -> None:
        if self.cfg.save != "y":
            return

        with StepTimer("7. Saving outputs"):
            prefix = f"{self.case.author}_{self.case.name}_{self.MN_pop}_"
            np.save(prefix + "time_array", self.time[self.range_start : self.range_stop], allow_pickle=True)
            np.save(prefix + "exp_force", self.Force[self.range_start : self.range_stop], allow_pickle=True)
            np.save(prefix + "exp_discharge_times", self.disch_times, allow_pickle=True)
            np.save(prefix + "PRED_discharge_times", self.Firing_times_sim, allow_pickle=True)

            parameters_payload = np.array(
                [
                    self.range_start,
                    self.range_stop,
                    self.t_start,
                    self.case.end_force,
                    self.Nb_MN,
                    self.Cm_rec,
                    self.cfg.cm_derec_calib,
                    self.cfg.adapt_kR,
                ],
                dtype=object,
            )
            
            np.save(prefix + "parameters", parameters_payload, allow_pickle=True)
            main_results_payload = np.array(
                [
                    self.Real_MN_pop,
                    self.Calib_sizes,
                    self.Cm_rec,
                    self.Cm_derec,
                    self.a_size,
                    self.c_size,
                    self.r2_size_calib,
                    self.kR_derec,
                    self.r2_exp,
                    self.nRMSE_exp,
                    self.r2_sim,
                    self.nRMSE_sim,
                ],
                dtype=object,
            )

            np.save(prefix + "MAIN_results", main_results_payload, allow_pickle=True)

            np.save(prefix + "calib_onset_error", self.Calib_delta_tf1, allow_pickle=True)
            np.save(prefix + "calib_nRMSE", self.Calib_RMS_table, allow_pickle=True)
            np.save(prefix + "calib_r2", self.Calib_r2_table, allow_pickle=True)
            np.save(prefix + "calib_FIDF", self.Calib_FIDF_list, allow_pickle=True)
