from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from .config import TEST_CASES, TestCase


@dataclass
class MainRunMetadata:
    author: str
    cm_calib: str
    cm_derec_calib: str
    adapt_kR: str
    cm_rec: float
    cm_derec: float
    a_size: float
    c_size: float
    r2_size_calib: float
    kR_derec: float
    r2_exp: float
    nRMSE_exp: float
    r2_sim: float
    nRMSE_sim: float


@dataclass
class ValidationOutputs:
    onset_error: Optional[np.ndarray] = None
    nRMSE: Optional[np.ndarray] = None
    r2: Optional[np.ndarray] = None
    size_distributions: Optional[np.ndarray] = None
    fidf_sim_test: Optional[np.ndarray] = None

    @property
    def available(self) -> bool:
        return self.onset_error is not None and self.nRMSE is not None and self.r2 is not None


@dataclass
class CalibrationOutputs:
    onset_error: Optional[np.ndarray] = None
    nRMSE: Optional[np.ndarray] = None
    r2: Optional[np.ndarray] = None
    fidf: Optional[np.ndarray] = None


@dataclass
class StoredMNModelResults:
    dataset: str
    case: TestCase
    result_dir: Path
    main_prefix: str
    validation_prefix: str
    time: np.ndarray
    force: np.ndarray
    Nb_MN: int
    MN_pop: int
    real_MN_pop: np.ndarray
    exp_disch_times: np.ndarray
    firing_times_sim: np.ndarray
    range_start: int
    range_stop: int
    t_start: float
    plateau_time1: float
    plateau_time2: float
    end_force: float
    fs: int
    calib_sizes: np.ndarray
    main: MainRunMetadata
    validation: ValidationOutputs
    calibration: CalibrationOutputs
    used_reconstructed_force: bool = False


def _load_npy(path: Path, required: bool = True, default: Any = None) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required result file not found: {path}")
        return default
    return np.load(path, allow_pickle=True)


def _find_prefix(result_dir: Path, author: str, dataset: str, suffix: str, preferred_mn_pop: Optional[int] = None) -> str:
    candidates = []
    if preferred_mn_pop is not None:
        candidates.append(f"{author}_{dataset}_{int(preferred_mn_pop)}_")
    candidates.append(f"{author}_{dataset}_32_")

    for prefix in candidates:
        if (result_dir / f"{prefix}{suffix}").exists():
            return prefix

    matches = sorted(result_dir.glob(f"{author}_{dataset}_*_{suffix}"))
    if matches:
        return matches[0].name[: -len(suffix)]

    raise FileNotFoundError(
        f"Could not find files matching {author}_{dataset}_*_{suffix} in {result_dir}. "
        f"Expected outputs produced by 1_MAIN_MN_model.py or the refactored main pipeline."
    )


def _find_validation_prefix(result_dir: Path, author: str, dataset: str) -> str:
    prefix = f"{author}_{dataset}_validation_"
    if (result_dir / f"{prefix}onset_error.npy").exists():
        return prefix
    matches = sorted(result_dir.glob(f"{author}_{dataset}_validation_*onset_error.npy"))
    if matches:
        return matches[0].name[: -len("onset_error.npy")]
    return prefix


def _parse_main_parameters(parameters: np.ndarray) -> Tuple[int, int, float, float, int, float, str, str]:
    # Original save format:
    # [range_start, range_stop, t_start, end_force, Nb_MN, Cm_rec, Cm_derec_calib, adapt_kR]
    values = list(np.asarray(parameters, dtype=object).ravel())
    if len(values) < 8:
        raise ValueError(f"Unexpected parameters array length {len(values)}; expected at least 8.")
    return (
        int(values[0]),
        int(values[1]),
        float(values[2]),
        float(values[3]),
        int(values[4]),
        float(values[5]),
        str(values[6]),
        str(values[7]),
    )


def _parse_main_results(main_results: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float, float, float, float, float, float]:
    # Original save format:
    # [Real_MN_pop, Calib_sizes, Cm_rec, Cm_derec, a_size, c_size, r2_size_calib,
    #  kR_derec, r2_exp, nRMSE_exp, r2_sim, nRMSE_sim]
    values = list(np.asarray(main_results, dtype=object).ravel())
    if len(values) < 12:
        raise ValueError(f"Unexpected MAIN_results array length {len(values)}; expected at least 12.")
    return (
        np.asarray(values[0]),
        np.asarray(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
        float(values[6]),
        float(values[7]),
        float(values[8]),
        float(values[9]),
        float(values[10]),
        float(values[11]),
    )


def _maybe_reconstruct_full_force_and_time(case: TestCase, repository_root: Path, fallback_time: np.ndarray, fallback_force: np.ndarray, nb_mn: int):
    """Try to rebuild the full preprocessed force/time arrays from Input_Exp_Data.

    The original plotting entry used a loader that returned full-length time/force
    arrays, although the main script saves only the simulated time window.  This
    helper restores that behavior when the raw .mat data are available.
    """
    input_dir = repository_root / "Input_Exp_Data"
    if not input_dir.exists():
        return fallback_time, fallback_force, False

    try:
        from EXP_DATA_PROCESSING_MOD import EXP_DATA_PROCESSING_func
        from Reshaping_MOD import preprocessing_func

        nb_raw, force_raw, _ = EXP_DATA_PROCESSING_func(case.author, case.name)
        force_full, time_full, *_ = preprocessing_func(
            case.author,
            force_raw,
            case.end_force,
            case.fs,
            int(nb_raw or nb_mn),
            case.plateau_time1,
            case.plateau_time2,
        )
        return np.asarray(time_full), np.asarray(force_full), True
    except Exception as exc:
        print(f"Could not reconstruct full force/time from Input_Exp_Data; using saved arrays. Reason: {exc}")
        return fallback_time, fallback_force, False


def load_stored_mn_model_results(
    dataset: str,
    results_dir: Path = Path("Results"),
    repository_root: Path = Path("."),
    mn_pop: Optional[int] = 32,
    require_validation: bool = False,
    reconstruct_full_force: bool = True,
) -> StoredMNModelResults:
    """Load outputs from the main and validation MN-model scripts.

    Parameters
    ----------
    dataset:
        Dataset name, e.g. ``TA_35_D`` or ``GM_30``.
    results_dir:
        Root results directory. Both ``Results`` and ``Results/<dataset>`` are
        accepted; the function chooses ``<dataset>`` subfolder when it exists.
    repository_root:
        Project root, used only for optional reconstruction of full time/force.
    mn_pop:
        Preferred pool size used in the main-result file prefix. If absent, the
        loader discovers the prefix automatically.
    require_validation:
        Raise if leave-one-out validation files are missing.
    reconstruct_full_force:
        Try to rebuild full preprocessed time/force from Input_Exp_Data.
    """
    if dataset not in TEST_CASES:
        available = ", ".join(TEST_CASES)
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {available}")
    case = TEST_CASES[dataset]
    repository_root = Path(repository_root).resolve()
    results_dir = Path(results_dir)
    if not results_dir.is_absolute():
        results_dir = repository_root / results_dir
    result_dir = results_dir / dataset if (results_dir / dataset).exists() else results_dir
    if not result_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {result_dir}")

    main_prefix = _find_prefix(result_dir, case.author, dataset, "time_array.npy", preferred_mn_pop=mn_pop)
    validation_prefix = _find_validation_prefix(result_dir, case.author, dataset)

    time_saved = np.asarray(_load_npy(result_dir / f"{main_prefix}time_array.npy"), dtype=float)
    force_saved = np.asarray(_load_npy(result_dir / f"{main_prefix}exp_force.npy"), dtype=float)
    exp_disch_times = _load_npy(result_dir / f"{main_prefix}exp_discharge_times.npy")
    firing_times_sim = _load_npy(result_dir / f"{main_prefix}PRED_discharge_times.npy")
    parameters = _load_npy(result_dir / f"{main_prefix}parameters.npy")
    main_results = _load_npy(result_dir / f"{main_prefix}MAIN_results.npy")

    range_start, range_stop, t_start, end_force, nb_mn, cm_rec_param, cm_derec_calib, adapt_kR = _parse_main_parameters(parameters)
    (
        real_MN_pop,
        calib_sizes,
        cm_rec,
        cm_derec,
        a_size,
        c_size,
        r2_size_calib,
        kR_derec,
        r2_exp,
        nRMSE_exp,
        r2_sim,
        nRMSE_sim,
    ) = _parse_main_results(main_results)

    mn_pop_loaded = len(firing_times_sim) if hasattr(firing_times_sim, "__len__") else int(mn_pop or 32)
    if mn_pop_loaded == 0:
        mn_pop_loaded = int(mn_pop or 32)

    if reconstruct_full_force:
        time, force, used_reconstructed = _maybe_reconstruct_full_force_and_time(case, repository_root, time_saved, force_saved, nb_mn)
    else:
        time, force, used_reconstructed = time_saved, force_saved, False
        range_start = 0
        range_stop = len(time_saved)

    main_meta = MainRunMetadata(
        author=case.author,
        cm_calib="unknown",
        cm_derec_calib=cm_derec_calib,
        adapt_kR=adapt_kR,
        cm_rec=float(cm_rec if np.isfinite(cm_rec) else cm_rec_param),
        cm_derec=float(cm_derec),
        a_size=float(a_size),
        c_size=float(c_size),
        r2_size_calib=float(r2_size_calib),
        kR_derec=float(kR_derec),
        r2_exp=float(r2_exp),
        nRMSE_exp=float(nRMSE_exp),
        r2_sim=float(r2_sim),
        nRMSE_sim=float(nRMSE_sim),
    )

    validation = ValidationOutputs(
        onset_error=_load_npy(result_dir / f"{validation_prefix}onset_error.npy", required=require_validation, default=None),
        nRMSE=_load_npy(result_dir / f"{validation_prefix}nRMSE.npy", required=require_validation, default=None),
        r2=_load_npy(result_dir / f"{validation_prefix}r2.npy", required=require_validation, default=None),
        size_distributions=_load_npy(result_dir / f"{validation_prefix}Size_distributions.npy", required=False, default=None),
        fidf_sim_test=_load_npy(result_dir / f"{validation_prefix}FIDF_sim_test.npy", required=False, default=None),
    )

    calibration = CalibrationOutputs(
        onset_error=_load_npy(result_dir / f"{main_prefix}calib_onset_error.npy", required=False, default=None),
        nRMSE=_load_npy(result_dir / f"{main_prefix}calib_nRMSE.npy", required=False, default=None),
        r2=_load_npy(result_dir / f"{main_prefix}calib_r2.npy", required=False, default=None),
        fidf=_load_npy(result_dir / f"{main_prefix}calib_FIDF.npy", required=False, default=None),
    )

    return StoredMNModelResults(
        dataset=dataset,
        case=case,
        result_dir=result_dir,
        main_prefix=main_prefix,
        validation_prefix=validation_prefix,
        time=np.asarray(time, dtype=float),
        force=np.asarray(force, dtype=float),
        Nb_MN=int(nb_mn),
        MN_pop=int(mn_pop_loaded),
        real_MN_pop=np.asarray(real_MN_pop),
        exp_disch_times=np.asarray(exp_disch_times, dtype=object),
        firing_times_sim=np.asarray(firing_times_sim, dtype=object),
        range_start=int(range_start),
        range_stop=int(min(range_stop, len(time))),
        t_start=float(t_start),
        plateau_time1=float(case.plateau_time1),
        plateau_time2=float(case.plateau_time2),
        end_force=float(end_force),
        fs=int(case.fs),
        calib_sizes=np.asarray(calib_sizes),
        main=main_meta,
        validation=validation,
        calibration=calibration,
        used_reconstructed_force=used_reconstructed,
    )
