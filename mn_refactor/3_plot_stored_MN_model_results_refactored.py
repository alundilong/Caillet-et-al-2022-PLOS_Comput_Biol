from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from mn_pipeline.results_plotting import ResultsPlotConfig, StoredResultsPlotter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot stored motoneuron-model outputs from the main and validation pipelines."
    )
    parser.add_argument(
        "--dataset",
        default="TA_35_D",
        choices=["TA_35_D", "TA_35_H", "TA_50", "GM_30"],
        help="Dataset/trial to plot.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Results"),
        help="Results root directory or dataset-specific results directory.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("."),
        help="Project root containing MN_Modules/ and optionally Input_Exp_Data/.",
    )
    parser.add_argument(
        "--mn-pop",
        type=int,
        default=32,
        help="Preferred MN-pool size used in the saved main-result prefix, e.g. 32.",
    )
    parser.add_argument(
        "--plot-fidfs",
        default="no",
        choices=["yes", "no", "y", "n"],
        help="Plot individual calibration/validation FIDFs. This can create many figures.",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        help="Plot only outputs from the main reconstruction pipeline.",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Plot only outputs from the leave-one-out validation pipeline.",
    )
    parser.add_argument(
        "--require-validation",
        action="store_true",
        help="Raise an error if validation .npy files are missing.",
    )
    parser.add_argument(
        "--no-reconstruct-full-force",
        action="store_true",
        help="Do not reconstruct full force/time from Input_Exp_Data; use saved arrays only.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Save every generated matplotlib figure automatically.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures_results"),
        help="Directory for saved figures when --save-figures is used.",
    )
    parser.add_argument(
        "--keep-figures-open",
        action="store_true",
        help="Keep figures open after saving instead of closing them automatically.",
    )
    return parser


def main() -> None:
    warnings.filterwarnings("ignore")
    args = build_parser().parse_args()

    if args.main_only and args.validation_only:
        raise ValueError("Choose at most one of --main-only or --validation-only.")

    include_main = not args.validation_only
    include_validation = not args.main_only

    cfg = ResultsPlotConfig(
        dataset=args.dataset,
        repository_root=args.repository_root,
        results_dir=args.results_dir,
        mn_pop=args.mn_pop,
        plot_fidfs=args.plot_fidfs,
        include_main=include_main,
        include_validation=include_validation,
        require_validation=args.require_validation,
        save_figures=args.save_figures,
        figures_dir=args.figures_dir,
        close_saved_figures=not args.keep_figures_open,
        reconstruct_full_force=not args.no_reconstruct_full_force,
    )
    StoredResultsPlotter(cfg).run()


if __name__ == "__main__":
    main()
