"""
Readable and compatibility-focused refactor of 1_MAIN_MN_model.py.

This script preserves the original scientific workflow and plotting behavior,
while organizing the code into explicit pipeline steps and replacing selected
helper modules with vectorized, drop-in-compatible implementations.

Run from the repository root that contains:
    MN_Modules/
    Input_Exp_Data/

Examples
--------
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --plot y
python 1_MAIN_MN_model_refactored.py --dataset TA_35_D --plot y --save-figures
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --plot n --save y --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mn_pipeline.config import RunConfig, TEST_CASES
from mn_pipeline.workflow import MotoneuronReconstructionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored MN-pool reconstruction workflow")
    parser.add_argument("--dataset", default="TA_35_D", choices=sorted(TEST_CASES.keys()))
    parser.add_argument("--repo-root", default=".", help="Repository root containing MN_Modules and Input_Exp_Data")
    parser.add_argument("--mn-pop", type=int, default=32, help="Virtual MN pool size to simulate")
    parser.add_argument("--true-mn-pop", type=int, default=None, help="True MN pool size for threshold mapping; default equals --mn-pop")
    parser.add_argument("--cm-derec-calib", choices=["yes", "no"], default="no")
    parser.add_argument("--cm-derec", type=float, default=2.0e-2)
    parser.add_argument("--adapt-kR", choices=["y", "n", "yes", "no"], default="y")
    parser.add_argument("--plot", choices=["y", "n", "yes", "no"], default="y")
    parser.add_argument("--save", choices=["y", "n", "yes", "no"], default="n")
    parser.add_argument("--nb-coherence-tests", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None, help="Set NumPy and Python random seed for reproducibility")
    parser.add_argument("--save-figures", action="store_true", help="Save every figure produced by the original PLOTS.py functions")
    parser.add_argument("--figures-dir", default="figures", help="Directory used when --save-figures is enabled")
    parser.add_argument("--keep-figures-open", action="store_true", help="Do not close figures after saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        dataset=args.dataset,
        repository_root=Path(args.repo_root),
        mn_pop=args.mn_pop,
        true_mn_pop=args.true_mn_pop,
        cm_derec_calib=args.cm_derec_calib,
        cm_derec=args.cm_derec,
        cm_derec_array=np.arange(1.6, 2.4, 0.2) * 1e-2,
        adapt_kR=args.adapt_kR,
        plot=args.plot,
        save=args.save,
        nb_coherence_tests=args.nb_coherence_tests,
        seed=args.seed,
        save_figures=args.save_figures,
        figures_dir=Path(args.figures_dir),
        close_saved_figures=not args.keep_figures_open,
    )
    pipeline = MotoneuronReconstructionPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
