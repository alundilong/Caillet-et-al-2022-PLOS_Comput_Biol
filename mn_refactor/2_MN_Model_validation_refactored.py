"""
Readable refactor of 2_MN_Model_validation.py.

This entry point performs leave-one-MN-out validation:
    1. Hold out one experimentally identified spike train.
    2. Use the remaining Nr-1 trains to compute common input, thresholds, ARP,
       and calibrated size distribution.
    3. Predict the held-out MN's size, ARP, spike train, and FIDF.
    4. Compare predicted vs experimental FIDF using onset error, nRMSE, nME, and r².

Run from the repository root that contains:
    MN_Modules/
    Input_Exp_Data/

Examples
--------
python 2_MN_Model_validation_refactored.py --dataset GM_30 --plot y --seed 1
python 2_MN_Model_validation_refactored.py --dataset TA_35_D --plot n --save y --seed 1
python 2_MN_Model_validation_refactored.py --dataset GM_30 --plot y --save-figures --figures-dir figs_validation_GM_30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mn_pipeline.validation import (
    VALIDATION_TEST_CASES,
    LeaveOneOutValidationPipeline,
    ValidationConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-MN-out validation workflow")
    parser.add_argument("--dataset", default="TA_35_D", choices=sorted(VALIDATION_TEST_CASES.keys()))
    parser.add_argument("--repo-root", default=".", help="Repository root containing MN_Modules and Input_Exp_Data")
    parser.add_argument("--mn-pop", type=int, default=None, help="Full MN pool size; default follows the validation test-case table")
    parser.add_argument("--cm-derec", type=float, default=2.0e-2, help="Cm_derec value obtained from 1_MAIN_MN_model sensitivity analysis")
    parser.add_argument("--adapt-kR", choices=["y", "n", "yes", "no"], default="y")
    parser.add_argument("--plot", choices=["y", "n", "yes", "no"], default="y")
    parser.add_argument("--save", choices=["y", "n", "yes", "no"], default="y")
    parser.add_argument("--seed", type=int, default=None, help="Set NumPy and Python random seed for reproducibility")
    parser.add_argument("--save-figures", action="store_true", help="Save every figure produced by PLOTS.py")
    parser.add_argument("--figures-dir", default="figures_validation", help="Directory used when --save-figures is enabled")
    parser.add_argument("--keep-figures-open", action="store_true", help="Do not close figures after saving")
    parser.add_argument(
        "--no-fold-detail-plots",
        action="store_true",
        help="Keep final/held-out FIDF plots but suppress per-fold ARP/current/size distribution plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ValidationConfig(
        dataset=args.dataset,
        repository_root=Path(args.repo_root),
        mn_pop=args.mn_pop,
        cm_derec=args.cm_derec,
        adapt_kR=args.adapt_kR,
        plot=args.plot,
        save=args.save,
        seed=args.seed,
        save_figures=args.save_figures,
        figures_dir=Path(args.figures_dir),
        close_saved_figures=not args.keep_figures_open,
        plot_fold_details=not args.no_fold_detail_plots,
    )
    LeaveOneOutValidationPipeline(cfg).run()


if __name__ == "__main__":
    main()
