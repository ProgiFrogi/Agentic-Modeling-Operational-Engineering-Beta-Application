"""Safe data inspection tools for agents (CSV under workspace)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def ensure_workspace_kaggle_csv_aliases(workspace_root: str) -> List[str]:
    """
    Many models assume data/train.csv. Kaggle zips usually put train.csv in the workspace root.
    Create data/train.csv and data/test.csv as symlinks to ../train.csv when missing (Unix-friendly).
    """
    root = Path(workspace_root).resolve()
    created: List[str] = []
    data_dir = root / "data"
    for name in ("train.csv", "test.csv"):
        src = root / name
        if not src.is_file():
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        alias = data_dir / name
        if alias.exists():
            continue
        try:
            rel = os.path.relpath(src, start=data_dir)
            os.symlink(rel, alias, target_is_directory=False)
            created.append(str(alias))
        except OSError:
            pass
    return created


_SKIP_PATH_PARTS = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules"})


def list_workspace_files(
    workspace_root: str,
    glob_pattern: str = "*",
    max_files: int = 500,
) -> Dict[str, Any]:
    """
    List files under the competition workspace (relative POSIX paths).
    Use glob_pattern '*.csv' for CSVs only; '*' for all files (capped).
    """
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        return {"ok": False, "error": "workspace is not a directory"}
    pat = (glob_pattern or "*").strip() or "*"
    paths: List[str] = []
    truncated = False
    try:
        gen = root.rglob("*") if pat in ("*", "**/*") else root.rglob(pat)
        for p in gen:
            if not p.is_file():
                continue
            if p.name.startswith("."):
                continue
            if _SKIP_PATH_PARTS.intersection(p.parts):
                continue
            try:
                rel = p.resolve().relative_to(root)
            except ValueError:
                continue
            paths.append(rel.as_posix())
            if len(paths) >= max_files:
                truncated = True
                break
        paths.sort()
        # Same inode via symlink (e.g. train.csv and data/train.csv) — one entry, prefer shorter path
        by_inode: Dict[int, str] = {}
        for rel in paths:
            full = (root / rel).resolve()
            try:
                ino = full.stat().st_ino
            except OSError:
                continue
            prev = by_inode.get(ino)
            if prev is None or len(rel) < len(prev):
                by_inode[ino] = rel
        paths = sorted(by_inode.values())
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "workspace": str(root),
        "glob_pattern": pat,
        "count": len(paths),
        "truncated": truncated,
        "files": paths,
    }


def _normalize_agent_csv_path(workspace_root: str, file_path: str) -> str:
    """
    Models often ask for data/train.csv; this competition stores train.csv in the workspace root.
    If the wrong path is missing but the root file exists, use the root file.
    """
    root = Path(workspace_root).resolve()
    raw = (file_path or "").strip()
    if not raw:
        return raw
    try:
        p = Path(raw)
        if p.is_absolute():
            p = p.resolve()
            try:
                rel = p.relative_to(root)
                raw = rel.as_posix()
            except ValueError:
                raw = p.as_posix()
    except OSError:
        pass
    key = raw.replace("\\", "/").lower().lstrip("/")
    wrong_right = (
        ("data/train.csv", "train.csv"),
        ("data/test.csv", "test.csv"),
    )
    for wrong, right in wrong_right:
        if key == wrong:
            wpath = root / wrong
            rpath = root / right
            if not wpath.is_file() and rpath.is_file():
                return right
            break
    return raw.replace("\\", "/")


def _resolve_under_workspace(workspace_root: str, file_path: str) -> Path:
    root = Path(workspace_root).resolve()
    file_path = _normalize_agent_csv_path(workspace_root, file_path)
    candidate = (root / file_path).resolve() if not os.path.isabs(file_path) else Path(file_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {file_path}") from exc
    return candidate


def check_file(workspace_root: str, file_path: str) -> Dict[str, Any]:
    """Return whether file exists under workspace_root."""
    try:
        p = _resolve_under_workspace(workspace_root, file_path)
        exists = p.is_file()
        return {"exists": exists, "resolved_path": str(p)}
    except ValueError as e:
        return {"exists": False, "error": str(e)}


def read_data(workspace_root: str, file_path: str, nrows: Optional[int] = 200) -> Dict[str, Any]:
    """
    Read a preview of a CSV (default 200 rows). Full path must stay under workspace_root.
    """
    try:
        p = _resolve_under_workspace(workspace_root, file_path)
        if not p.is_file():
            hint = ""
            if "data" + os.sep + "train" in str(p).replace("\\", "/") or str(p).endswith(
                ("data/train.csv", "data\\train.csv")
            ):
                hint = " This dataset uses train.csv in the workspace root, not data/train.csv."
            return {"ok": False, "error": f"File not found: {p}.{hint}"}
        df = pd.read_csv(p, nrows=nrows)
        return {
            "ok": True,
            "path": str(p),
            "shape_preview": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "head_csv": df.head(15).to_csv(index=False),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def inspect_data(workspace_root: str, file_path: str) -> Dict[str, Any]:
    """Summary stats: shape, dtypes, null counts, numeric describe sample."""
    try:
        p = _resolve_under_workspace(workspace_root, file_path)
        if not p.is_file():
            hint = ""
            if "data/train" in str(p).replace("\\", "/"):
                hint = " Use train.csv in the workspace root."
            return {"ok": False, "error": f"File not found: {p}.{hint}"}
        df = pd.read_csv(p)
        nulls = df.isnull().sum().to_dict()
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        describe = {}
        if numeric_cols:
            describe = df[numeric_cols].describe().to_dict()
        return {
            "ok": True,
            "path": str(p),
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "null_counts": {k: int(v) for k, v in nulls.items() if v > 0},
            "numeric_summary": describe,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
