from __future__ import annotations

import numpy as np


class CurrentInput:
    """
    Fast callable current input I(t) backed by a precomputed array.

    It reproduces the original logic:
        I(t)=0 outside [first recruitment, last derecruitment]
        I(t)=I1 + G * common_input[int(t*fs)] inside the active window.
    """

    def __init__(self, common_signal, thresholds, G: float, I1: float, fs: int = 2048):
        self.common_signal = np.asarray(common_signal, dtype=float)
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.G = float(G)
        self.I1 = float(I1)
        self.fs = int(fs)
        self.t_on = float(self.thresholds[0, 0])
        self.t_off = float(np.max(self.thresholds[:, 3]))
        self.values = self.I1 + self.G * self.common_signal
        mask = np.ones_like(self.values, dtype=bool)
        idx_on = max(0, int(self.t_on * self.fs))
        idx_off = min(len(self.values), int(self.t_off * self.fs) + 1)
        mask[:idx_on] = False
        mask[idx_off:] = False
        self.values = np.where(mask, self.values, 0.0)
        # Original code returns 0 when common time is outside thresholds. The
        # explicit time guard below also handles non-sample-aligned t.

    def __call__(self, t: float) -> float:
        if t < self.t_on or t > self.t_off:
            return 0.0
        idx = int(t * self.fs)
        if idx < 0 or idx >= len(self.values):
            return 0.0
        return float(self.values[idx])

    def as_array(self):
        return self.values.copy()

    def as_nA(self):
        return self.values * 1e9
