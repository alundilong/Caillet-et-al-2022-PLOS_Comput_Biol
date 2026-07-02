# Refactored motoneuron-pool reconstruction workflow

This package refactors the uploaded `1_MAIN_MN_model.py` workflow while keeping the original scientific call sequence intact.

## What this refactor changes

The original script is a single procedural file. This refactor turns it into an explicit pipeline:

1. Load and preprocess experimental HDEMG/MN discharge data.
2. Compute experimental CST, common input, common control, and common noise.
3. Compute subset coherence, experimental thresholds, and MN locations in the real pool.
4. Compute experimental IDF/FIDF and inert-period / ARP distribution.
5. Compute the common synaptic current input `I(t)`.
6. Calibrate identified MN sizes.
7. Optionally perform `Cm_derec` sensitivity analysis.
8. Fit the calibrated size distribution back to the complete virtual MN pool.
9. Simulate the virtual MN pool with the original LIF solver.
10. Validate simulated common control against the force trace.
11. Save the same core outputs as the original script when requested.

## Why results should be much closer than the previous refactor

This version does **not** replace the scientific model with approximations. It still calls the same original modules for the core steps, including:

- `EXP_DATA_PROCESSING_MOD`
- `Reshaping_MOD`
- `EXP_THRESHOLDS_MOD`
- `IDF_MOD`
- `exp_ARP_MOD`
- `ARP_distrib_MOD`
- `Curr_input_MOD`
- `Simplified_Size_calibration_MOD`
- `Cm_derec_sensitivity_MOD`
- `calibrated_sizes_MOD`
- `run_LIF_simulation_MOD`
- `RMS_func`
- `PLOTS`

Selected helper modules were replaced by drop-in-compatible optimized versions:

- `CST_MOD.py`: vectorized binary spike matrix and CST construction.
- `But_filter_MOD.py`: cached Butterworth filter coefficients.
- `subset_coher_MOD.py`: vectorized subset summation.
- `MN_distirbution_MOD.py`: vectorized nearest-threshold mapping.
- `Hanning_filter_MOD.py`: vectorized multi-MU convolution when possible.
- `Virtual_pop_size_arp_MOD.py`: vectorized virtual size/ARP reconstruction.

The current-input function `I(t)` is backed by a precomputed array for faster repeated calls from the LIF solver, but it follows the same logic as the original code.

## Important reproducibility note

The original LIF solver uses random ARP variability through `np.random.normal`. The original script does not set a random seed, so repeated runs can differ even without refactoring. For fair comparison, run both workflows with a fixed seed. This refactored version supports:

```bash
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --seed 1
```

To reproduce exactly against the original, add equivalent `np.random.seed(1)` and `random.seed(1)` near the top of the original script before running it.

## Installation / placement

Place this folder at the repository root where `Input_Exp_Data/` is available:

```text
project_root/
├── 1_MAIN_MN_model_refactored.py
├── mn_pipeline/
├── MN_Modules/
└── Input_Exp_Data/
```

The uploaded `MN_Modules.zip` did not contain `MN_properties_relationships_MOD.py`, although several modules import it. A reconstructed copy based on the public repository formula definitions is included in `MN_Modules/`.

## Basic run

```bash
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --plot y
```

## Save all figures produced by the original plotting functions

```bash
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --plot y --save-figures --figures-dir figs_GM_30
```

This works by intercepting every `matplotlib.pyplot.show()` call from `PLOTS.py` and saving the open figure before closing it.

## Run without plots

```bash
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --plot n
```

Unlike the original script, this works because `I_smooth_list` is computed directly instead of being accidentally dependent on `plot_current_input()`.

## Save output `.npy` files

```bash
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --plot n --save y --seed 1
```

The saved filenames follow the original naming pattern:

```text
<author>_<dataset>_<MN_pop>_time_array.npy
<author>_<dataset>_<MN_pop>_exp_force.npy
<author>_<dataset>_<MN_pop>_exp_discharge_times.npy
<author>_<dataset>_<MN_pop>_PRED_discharge_times.npy
<author>_<dataset>_<MN_pop>_parameters.npy
<author>_<dataset>_<MN_pop>_MAIN_results.npy
<author>_<dataset>_<MN_pop>_calib_onset_error.npy
<author>_<dataset>_<MN_pop>_calib_nRMSE.npy
<author>_<dataset>_<MN_pop>_calib_r2.npy
<author>_<dataset>_<MN_pop>_calib_FIDF.npy
```

## Compare original and refactored saved outputs

After running both versions with the same seed and same settings:

```bash
python compare_original_vs_refactored.py --original original_outputs --refactored refactored_outputs
```

## Why some small differences can remain

Small differences can still occur because:

1. The LIF solver is stochastic unless the random seed is fixed.
2. Plotting functions sometimes recompute simulated FIDFs internally; these can consume additional random draws when plots are enabled.
3. The original code has a few plot-dependent side effects, especially around `I_smooth_list`. This refactor removes that side effect to allow `--plot n` to run.
4. Some helper functions now return numeric arrays instead of object arrays where safe. Downstream behavior should be compatible, but exact object dtype comparisons may differ.

For strict numerical comparison, use:

```bash
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --plot n --save y --seed 1
```

and use the same no-plot, fixed-seed condition for the original script.
