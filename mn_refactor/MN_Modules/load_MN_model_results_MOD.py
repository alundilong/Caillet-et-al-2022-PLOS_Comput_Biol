"""Compatibility loader for the original 3_plot_stored_MN_model_results.py API.

The refactored code uses :mod:`mn_pipeline.results_loader`. This module keeps
``load_MN_model_results`` available for old scripts that still import it from
``MN_Modules``.
"""

from __future__ import annotations

from pathlib import Path

from mn_pipeline.results_loader import load_stored_mn_model_results


def load_MN_model_results(test, path_to_data):
    result = load_stored_mn_model_results(
        dataset=test,
        results_dir=Path(path_to_data),
        repository_root=Path("."),
        mn_pop=32,
        require_validation=False,
        reconstruct_full_force=True,
    )
    metadata = [
        result.main.author,
        result.main.cm_calib,
        result.main.cm_derec_calib,
        result.main.adapt_kR,
        result.main.cm_rec,
        result.main.cm_derec,
        result.main.a_size,
        result.main.c_size,
        result.main.r2_size_calib,
        result.main.kR_derec,
        result.main.r2_exp,
        result.main.nRMSE_exp,
        result.main.r2_sim,
        result.main.nRMSE_sim,
    ]
    validation = [
        result.validation.onset_error,
        result.validation.nRMSE,
        result.validation.r2,
        result.validation.size_distributions,
        result.validation.fidf_sim_test,
    ]
    calibration = [
        result.calibration.onset_error,
        result.calibration.nRMSE,
        result.calibration.r2,
        result.calibration.fidf,
    ]
    return (
        result.time,
        result.case.muscle,
        result.case.mvc,
        result.force,
        result.Nb_MN,
        result.MN_pop,
        result.real_MN_pop,
        result.exp_disch_times,
        result.firing_times_sim,
        result.range_start,
        result.range_stop,
        result.t_start,
        result.plateau_time1,
        result.plateau_time2,
        result.end_force,
        result.fs,
        result.calib_sizes,
        metadata,
        validation,
        calibration,
    )
