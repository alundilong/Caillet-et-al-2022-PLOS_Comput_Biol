from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np


@dataclass(frozen=True)
class TestCase:
    author: str
    name: str
    muscle: str
    plateau_time1: float
    plateau_time2: float
    end_force: float
    mvc: float
    mn_pop: int
    fs: int


TEST_CASES: Dict[str, TestCase] = {
    "TA_35_D": TestCase("DelVecchio", "TA_35_D", "TA", 10.3, 20.5, 30.0, 0.35, 400, 2048),
    "TA_35_H": TestCase("Hug", "TA_35_H", "TA", 10.5, 20.5, 30.0, 0.35, 400, 2048),
    "TA_50": TestCase("Hug", "TA_50", "TA", 12.0, 21.8, 34.5, 0.50, 400, 2048),
    "GM_30": TestCase("Hug", "GM_30", "GM", 9.1, 19.1, 33.5, 0.30, 400, 2048),
}


@dataclass
class RunConfig:
    dataset: str = "TA_35_D"
    mn_pop: int = 32
    true_mn_pop: Optional[int] = None
    cm_derec_calib: str = "no"
    cm_derec: float = 2.0e-2
    cm_derec_array: Optional[np.ndarray] = None
    adapt_kR: str = "y"
    plot: str = "y"
    save: str = "n"
    nb_coherence_tests: int = 20
    seed: Optional[int] = None
    repository_root: Path = Path(".")
    save_figures: bool = False
    figures_dir: Path = Path("figures")
    close_saved_figures: bool = True

    def __post_init__(self) -> None:
        self.plot = _normalize_yes_no(self.plot, yes="y", no="n")
        self.save = _normalize_yes_no(self.save, yes="y", no="n")
        self.adapt_kR = _normalize_yes_no(self.adapt_kR, yes="y", no="n")
        if self.cm_derec_calib not in {"yes", "no"}:
            raise ValueError("cm_derec_calib must be 'yes' or 'no'")
        if self.cm_derec_array is None:
            self.cm_derec_array = np.arange(1.6, 2.4, 0.2) * 1e-2
        if self.true_mn_pop is None:
            self.true_mn_pop = int(self.mn_pop)

    @property
    def test_case(self) -> TestCase:
        try:
            return TEST_CASES[self.dataset]
        except KeyError as exc:
            available = ", ".join(TEST_CASES)
            raise ValueError(f"Unknown dataset {self.dataset!r}. Available datasets: {available}") from exc


def _normalize_yes_no(value: str, yes: str, no: str) -> str:
    value = str(value).strip().lower()
    if value in {"y", "yes", "true", "1"}:
        return yes
    if value in {"n", "no", "false", "0"}:
        return no
    raise ValueError(f"Expected yes/no value, got {value!r}")
