"""
Compare saved outputs from the original and refactored scripts.

Usage pattern:
1. Run the original script with save='y'. Move its output files into a folder,
   for example original_outputs/.
2. Run the refactored script with --save y and the same dataset/pool/seed.
   Move its output files into refactored_outputs/.
3. Run:
   python compare_original_vs_refactored.py --original original_outputs --refactored refactored_outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def load_npy_files(folder: Path):
    return {p.name: np.load(p, allow_pickle=True) for p in folder.glob("*.npy")}


def compare_arrays(a, b):
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
        if aa.shape != bb.shape:
            return f"shape differs: {aa.shape} vs {bb.shape}"
        max_abs = np.nanmax(np.abs(aa - bb)) if aa.size else 0.0
        return f"max_abs_diff={max_abs:.6e}"
    except Exception:
        return "object/non-numeric array; manual inspection needed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--refactored", required=True)
    args = parser.parse_args()

    original = load_npy_files(Path(args.original))
    refactored = load_npy_files(Path(args.refactored))

    all_names = sorted(set(original) | set(refactored))
    for name in all_names:
        if name not in original:
            print(f"ONLY refactored: {name}")
            continue
        if name not in refactored:
            print(f"ONLY original:   {name}")
            continue
        print(f"{name:35s} {compare_arrays(original[name], refactored[name])}")


if __name__ == "__main__":
    main()
