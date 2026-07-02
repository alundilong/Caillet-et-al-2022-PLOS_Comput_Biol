"""Regression utilities used by the MN reconstruction pipeline."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import curve_fit


def func_Lin(x, a):
    return a * x


def func_aff(x, a, b):
    return a * x + b


def func_quadra(x, a, b, c):
    return a * x**2 + b * x + c


def func_cubic(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d


def func_power(x, a, b):
    return a * x**b


def _size_power_factory(muscle: str, MN_pop: int) -> Callable:
    if muscle in {"TA", "GM"}:
        denominator = float(MN_pop)
    else:
        denominator = 550.0

    def size_power(x, a, c):
        return a * 2.4 ** (((x + 1) / denominator) ** c)

    return size_power


def _threshold_power_factory(muscle: str, MN_pop: int) -> Callable:
    if muscle in {"TA", "GM"}:
        denominator = float(MN_pop)
    else:
        denominator = 550.0

    def threshold_power(x, a, b, k):
        return k * (a * (x + 1) / denominator + 90 ** (((x + 1) / denominator) ** b))

    return threshold_power


def _select_function(fun: str, muscle: str, MN_pop: int) -> Callable:
    functions = {
        "lin": func_Lin,
        "aff": func_aff,
        "quadra": func_quadra,
        "cubic": func_cubic,
        "power": func_power,
        "size": _size_power_factory(muscle, MN_pop),
        "threshold": _threshold_power_factory(muscle, MN_pop),
    }
    if fun not in functions:
        raise ValueError(f"Unknown regression type: {fun!r}. Options: {sorted(functions)}")
    return functions[fun]


def regression(X, Y, fun, muscle, MN_pop):
    """Fit a named model and return ``(parameters, r_squared)``."""
    x = np.asarray(X, dtype=float).ravel()
    y = np.asarray(Y, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("Regression received no finite points.")

    func = _select_function(str(fun), str(muscle), int(MN_pop))
    popt, _ = curve_fit(func, x, y, maxfev=10000)
    residuals = y - func(x, *popt)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return popt, float(r_squared)
