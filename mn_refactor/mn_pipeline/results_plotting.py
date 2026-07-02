from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from .paths import repository_context
from .plotting import capture_matplotlib_show
from .results_loader import StoredMNModelResults, load_stored_mn_model_results


@dataclass
class ResultsPlotConfig:
    dataset: str = "TA_35_D"
    repository_root: Path = Path(".")
    results_dir: Path = Path("Results")
    mn_pop: int = 32
    plot_fidfs: str = "no"
    include_main: bool = True
    include_validation: bool = True
    require_validation: bool = False
    save_figures: bool = False
    figures_dir: Path = Path("figures_results")
    close_saved_figures: bool = True
    reconstruct_full_force: bool = True

    def __post_init__(self):
        self.plot_fidfs = _normalize_yes_no_word(self.plot_fidfs)


def _normalize_yes_no_word(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"y", "yes", "true", "1"}:
        return "yes"
    if value in {"n", "no", "false", "0"}:
        return "no"
    raise ValueError(f"Expected yes/no value, got {value!r}")


class StoredResultsPlotter:
    """Plot results already saved by the main and validation pipelines."""

    def __init__(self, config: ResultsPlotConfig):
        self.cfg = config
        self.results: Optional[StoredMNModelResults] = None
        self.signals: Dict[str, Any] = {}

    def run(self) -> StoredMNModelResults:
        with repository_context(self.cfg.repository_root):
            with capture_matplotlib_show(
                enabled=self.cfg.save_figures,
                figures_dir=Path(self.cfg.repository_root) / self.cfg.figures_dir,
                close=self.cfg.close_saved_figures,
            ):
                self.results = load_stored_mn_model_results(
                    dataset=self.cfg.dataset,
                    results_dir=self.cfg.results_dir,
                    repository_root=self.cfg.repository_root,
                    mn_pop=self.cfg.mn_pop,
                    require_validation=self.cfg.require_validation,
                    reconstruct_full_force=self.cfg.reconstruct_full_force,
                )
                self._print_loaded_summary()
                self._compute_common_inputs()
                self._plot_experimental_overview()
                if self.cfg.include_main:
                    self._plot_main_outputs()
                if self.cfg.include_validation:
                    self._plot_validation_outputs()
                return self.results

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def _compute_common_inputs(self) -> None:
        from CST_MOD import CST_func
        from But_filter_MOD import But_filter_func

        r = self.results
        assert r is not None
        print("Computing experimental and simulated CST/common-input signals...")
        binary_exp, cst_exp = CST_func(r.Nb_MN, r.time, r.exp_disch_times, fs=r.fs)
        common_control_exp = But_filter_func(4, cst_exp, fs=r.fs)
        common_input_exp = But_filter_func(10, cst_exp, fs=r.fs)

        binary_sim, cst_sim = CST_func(
            r.MN_pop,
            r.time[r.range_start : r.range_stop],
            r.firing_times_sim,
            unit="sec",
            fs=r.fs,
        )
        common_control_sim = But_filter_func(4, cst_sim, fs=r.fs)
        common_input_sim = But_filter_func(10, cst_sim, fs=r.fs)

        self.signals.update(
            binary_exp=binary_exp,
            cst_exp=cst_exp,
            common_control_exp=common_control_exp,
            common_input_exp=common_input_exp,
            binary_sim=binary_sim,
            cst_sim=cst_sim,
            common_control_sim=common_control_sim,
            common_input_sim=common_input_sim,
        )

    # ------------------------------------------------------------------
    # Plot sections
    # ------------------------------------------------------------------
    def _plot_experimental_overview(self) -> None:
        import PLOTS as plots

        r = self.results
        assert r is not None
        plots.plot_spike_trains_force_CI_func(
            r.Nb_MN,
            r.force,
            r.time,
            self.signals["common_input_exp"],
            self.signals["common_control_exp"],
            r.t_start,
            r.end_force,
            r.fs,
            r.exp_disch_times,
            r.case.mvc,
        )

    def _plot_main_outputs(self) -> None:
        import PLOTS as plots
        from IDF_MOD import IDF_func
        from Hanning_filter_MOD import Hanning_filter_func

        r = self.results
        assert r is not None

        print("OUTPUTS FROM 1_MAIN_MN_model.py / refactored main pipeline")
        print(f"The sensitivity analysis returned/used Cm_derec = {r.main.cm_derec}")
        print(f"The trendline fitted to calibrated sizes returned a = {r.main.a_size}, c = {r.main.c_size}")

        if r.calibration.onset_error is not None and r.calibration.nRMSE is not None and r.calibration.r2 is not None:
            plots.plot_achieved_error_S_calibration(
                r.Nb_MN,
                r.calib_sizes,
                r.calibration.onset_error,
                r.calibration.nRMSE,
                r.calibration.r2,
            )
        else:
            print("Calibration-error arrays not found; skipping calibration summary plot.")

        fidf_exp = None
        if self.cfg.plot_fidfs == "yes":
            print("Computing experimental FIDFs for optional calibration plots...")
            idf_dt, ff_full = IDF_func(r.Nb_MN, r.time, r.exp_disch_times, r.t_start, r.end_force, fs=r.fs)
            fidf_exp = Hanning_filter_func(r.Nb_MN, ff_full, fs=r.fs)
            if r.calibration.fidf is not None:
                for i in range(r.Nb_MN):
                    try:
                        plots.plot_exp_vs_sim_FIDF_func(
                            r.time,
                            r.range_start,
                            fidf_exp[i],
                            r.calibration.fidf[i],
                            i,
                            "calibration",
                        )
                    except Exception as exc:
                        print(f"Skipping calibration FIDF plot for MN {i + 1}: {exc}")
            else:
                print("Calibration FIDF array not found; skipping calibration FIDF plots.")

        def size_power(x, a, c):
            return a * 2.4 ** (((x + 1) / r.MN_pop) ** c)

        plots.plot_size_distribution_func(
            r.real_MN_pop,
            size_power,
            np.array([r.main.a_size, r.main.c_size]),
            r.main.a_size,
            r.main.c_size,
            r.main.r2_size_calib,
            r.calib_sizes,
            r.MN_pop,
        )

        plots.plot_spike_trains_force_CI_func(
            r.MN_pop,
            r.force,
            r.time,
            self.signals["common_input_sim"],
            self.signals["common_control_sim"],
            r.t_start,
            r.end_force,
            r.fs,
            r.firing_times_sim,
            r.case.mvc,
            "sim",
        )
        plots.plot_onionskin_representation_func(r.time, r.firing_times_sim, r.t_start, r.end_force, r.MN_pop, fs=r.fs)

        print("Global validation between normalized force trace and neural drive:")
        print(f"Coef of determination between EXP force trace and common control = {r.main.r2_exp}")
        print(f"nRMSE EXP = {r.main.nRMSE_exp}")
        print(f"Coef of determination between SIM force trace and common control = {r.main.r2_sim}")
        print(f"nRMSE SIM = {r.main.nRMSE_sim}")
        print("#-----------------------------------------------------------------------")

        plots.plot_final_results_func(
            r.Nb_MN,
            r.time,
            r.t_start,
            r.range_start,
            r.end_force,
            r.range_stop,
            self.signals["common_control_exp"],
            self.signals["common_control_sim"],
            self.signals["common_input_exp"],
            self.signals["common_input_sim"],
            r.force,
        )

    def _plot_validation_outputs(self) -> None:
        import PLOTS as plots
        from IDF_MOD import IDF_func
        from Hanning_filter_MOD import Hanning_filter_func

        r = self.results
        assert r is not None

        print("OUTPUTS FROM 2_MN_Model_validation.py / refactored validation pipeline")
        if not r.validation.available:
            print("Validation result arrays are not available; skipping validation plots.")
            if self.cfg.require_validation:
                raise FileNotFoundError("Validation arrays were requested but not found.")
            return

        nme_placeholder = "n"
        if r.validation.size_distributions is None:
            size_distrib = np.zeros((len(r.validation.nRMSE), 2))
        else:
            size_distrib = r.validation.size_distributions

        plots.plot_final_results_validation_func(
            r.Nb_MN - 1,
            r.validation.onset_error + 0.08,
            nme_placeholder,
            r.validation.nRMSE,
            r.validation.r2,
            size_distrib,
        )

        if self.cfg.plot_fidfs == "yes":
            print("Computing experimental FIDFs for optional validation plots...")
            idf_dt, ff_full = IDF_func(r.Nb_MN, r.time, r.exp_disch_times, r.t_start, r.end_force, fs=r.fs)
            fidf_exp = Hanning_filter_func(r.Nb_MN, ff_full, fs=r.fs)
            if r.validation.fidf_sim_test is not None:
                for i in range(r.Nb_MN):
                    try:
                        plots.plot_exp_vs_sim_FIDF_func(
                            r.time,
                            r.range_start,
                            fidf_exp[i],
                            r.validation.fidf_sim_test[i],
                            i,
                            "validation",
                        )
                    except Exception as exc:
                        print(f"Skipping validation FIDF plot for MN {i + 1}: {exc}")
            else:
                print("Validation FIDF array not found; skipping validation FIDF plots.")

        onset = np.asarray(r.validation.onset_error, dtype=float)
        nrmse = np.asarray(r.validation.nRMSE, dtype=float)
        r2 = np.asarray(r.validation.r2, dtype=float)
        print("Local validation between experimental and simulated FIDFs:")
        valid_onset = onset[onset > -1]
        valid_r2 = r2[r2 > 0.4]
        print(f"Average delta_ft1 = {np.nanmean(valid_onset) if valid_onset.size else np.nan}")
        print(f"Average nRMSE = {np.nanmean(nrmse)}")
        print(f"Average r2 = {np.nanmean(valid_r2) if valid_r2.size else np.nan}")
        if r.validation.size_distributions is not None:
            print(f"Average power = {np.nanmean(np.asarray(r.validation.size_distributions)[:, 1])}")
            print(f"Average min Size = {np.nanmean(np.asarray(r.validation.size_distributions)[:, 0])}")
        print("#-----------------------------------------------------------------------")

    def _print_loaded_summary(self) -> None:
        r = self.results
        assert r is not None
        print(f"Dataset: {r.dataset}")
        print(f"Results directory: {r.result_dir}")
        print(f"Main result prefix: {r.main_prefix}")
        print(f"Validation prefix: {r.validation_prefix}")
        print(f"Recorded MNs: {r.Nb_MN}")
        print(f"Simulated MN pool: {r.MN_pop}")
        print(f"Was Cm_derec calibrated? {r.main.cm_derec_calib}")
        print(f"Was kR adapted for derecruitment? {r.main.adapt_kR}")
        if r.used_reconstructed_force:
            print("Full force/time were reconstructed from Input_Exp_Data for plotting.")
        else:
            print("Using saved time/force arrays only. Some plots may show the simulated window only.")
        print("#-----------------------------------------------------------------------")
