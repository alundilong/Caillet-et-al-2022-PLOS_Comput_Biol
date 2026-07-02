# Full `MN_Modules` refactor

This folder is a drop-in refactor of the original `MN_Modules` directory for the Caillet et al. motoneuron-pool reconstruction workflow.

## Design goals

1. Keep the original public function names and signatures wherever the main workflow depends on them.
2. Preserve the original scientific workflow and numerical logic as much as possible.
3. Improve readability with clearer function boundaries, docstrings, validation, and typed local variables.
4. Improve computational efficiency in safe places:
   - cached Butterworth filter coefficients;
   - vectorized CST construction;
   - vectorized threshold and MN-pool mapping;
   - vectorized Hanning filtering for rectangular arrays;
   - faster LIF solver internals by replacing repeated `np.append` and pre-evaluating current input when possible;
   - cleaner repeated array conversions and safer correlation/RMS helpers.

## Important compatibility notes

The LIF solver contains stochastic ARP perturbation, just like the original code. Use a fixed seed in the top-level script for run-to-run comparison.

The plotting module is intentionally preserved close to the original so the same figures remain available. The computational modules were refactored more aggressively than `PLOTS.py`.

## Recommended workflow

Run the original and refactored workflows on the same dataset with the same random seed, then compare saved outputs:

```bash
python 1_MAIN_MN_model_refactored.py --dataset GM_30 --plot n --save y --seed 1
```

If exact values differ, check the stochastic LIF step first. For strict regression testing, compare intermediate arrays step-by-step:

1. `CST_exp`
2. `common_input_exp`, `common_control_exp`
3. `THRESHOLDS`
4. `Real_MN_pop`
5. `FIDF_exp`
6. `ARP_table`
7. `I_array`
8. `Calib_sizes`
9. `Virtual_size_arr`, `Virtual_ARP_arr`
10. `Firing_times_sim`

## Main modules refactored

- `EXP_DATA_PROCESSING_MOD.py`: robust `.mat` loading and sorting by first discharge.
- `Reshaping_MOD.py`: explicit force trimming and time construction.
- `CST_MOD.py`: vectorized binary matrix and CST creation.
- `But_filter_MOD.py`: cached filter design.
- `EXP_THRESHOLDS_MOD.py`: clearer threshold matrix construction.
- `MN_distirbution_MOD.py`: vectorized nearest-threshold mapping.
- `IDF_MOD.py`: clearer IDF and impulse-train construction.
- `Hanning_filter_MOD.py`: cached window and vectorized filtering when possible.
- `RC_LIF_MOD.py`: same LIF recurrence, faster storage and current evaluation.
- `FF_filt_func.py`: simplified single-MN simulation-to-FIDF path.
- `Simplified_Size_calibration_MOD.py`: clearer per-MN calibration with robust metrics.
- `run_LIF_simulation_MOD.py`: clearer final pool simulation loop.

