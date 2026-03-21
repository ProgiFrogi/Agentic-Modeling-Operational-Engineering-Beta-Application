"""Remove agent scratch files so the next run does not import stale modules."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def clean_agent_artifacts(workspace_root: str) -> Dict[str, Any]:
    """
    Delete typical multi-run clutter:
    - scripts/_agent_*.py (auto paths from tool_save_code)
    - tmp*.py in workspace root (execute_code temp scripts, if any remain)
    - all __pycache__ under the workspace
    Does not remove train.csv, data/*.py, or solution.csv.
    """
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        return {"ok": False, "error": "not a directory", "removed": []}
    removed: List[str] = []

    scripts = root / "scripts"
    if scripts.is_dir():
        for p in scripts.glob("_agent_*.py"):
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                pass

    for p in root.glob("tmp*.py"):
        if p.is_file():
            try:
                p.unlink()
                removed.append(str(p))
            except OSError:
                pass

    for p in root.rglob("__pycache__"):
        if p.is_dir():
            try:
                shutil.rmtree(p, ignore_errors=True)
                removed.append(str(p))
            except OSError:
                pass

    return {"ok": True, "removed_count": len(removed), "removed": removed[:200]}


def fork_workspace_with_data_symlinks(base_workspace: str) -> str:
    """
    Create sibling directory ``<base>_<YYYYMMDD_HHMMSS>`` and symlink competition CSVs
    from base_workspace so each run starts without old scripts/pycache.
    """
    base = Path(base_workspace).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"base workspace missing: {base}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = base.parent / f"{base.name}_{stamp}_{os.getpid()}"
    dest.mkdir(parents=True, exist_ok=False)

    linked = 0
    for name in (
        "train.csv",
        "test.csv",
        "sample_submition.csv",
        "sample_submission.csv",
        "description.md",
    ):
        src = base / name
        if not src.is_file():
            continue
        link = dest / name
        try:
            os.symlink(src, link, target_is_directory=False)
        except OSError:
            shutil.copy2(src, link)
        linked += 1

    if linked == 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise FileNotFoundError(
            f"no train.csv (or other competition files) in base workspace: {base}"
        )

    return str(dest)
