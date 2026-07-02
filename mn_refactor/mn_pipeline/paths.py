from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path


@contextlib.contextmanager
def repository_context(repository_root: Path):
    """Temporarily execute from repository root and expose MN_Modules on sys.path."""
    repository_root = Path(repository_root).resolve()
    modules_dir = repository_root / "MN_Modules"
    if not modules_dir.exists():
        raise FileNotFoundError(f"MN_Modules directory not found: {modules_dir}")

    old_cwd = Path.cwd()
    inserted = []
    for p in (str(modules_dir), str(repository_root)):
        if p not in sys.path:
            sys.path.insert(0, p)
            inserted.append(p)
    os.chdir(repository_root)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        for p in inserted:
            try:
                sys.path.remove(p)
            except ValueError:
                pass
