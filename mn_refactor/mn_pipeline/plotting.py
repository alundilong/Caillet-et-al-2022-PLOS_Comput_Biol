from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator


@contextlib.contextmanager
def capture_matplotlib_show(enabled: bool, figures_dir: Path, close: bool = True) -> Iterator[None]:
    """
    Save every figure produced by plt.show().

    The original code produces many figures through direct plt.show() calls in
    PLOTS.py. This context preserves those calls while optionally saving every
    open figure before it is shown/closed.
    """
    if not enabled:
        yield
        return

    import matplotlib.pyplot as plt

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    original_show = plt.show
    counter = {"n": 0}

    def show_and_save(*args, **kwargs):
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            counter["n"] += 1
            fig.savefig(figures_dir / f"figure_{counter['n']:03d}.png", dpi=180, bbox_inches="tight")
        if close:
            plt.close("all")
        else:
            return original_show(*args, **kwargs)

    plt.show = show_and_save
    try:
        yield
    finally:
        plt.show = original_show
