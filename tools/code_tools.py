"""Validate and execute Python in an isolated subprocess under a workspace directory."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


def _resolve_under_workspace(workspace_root: str, file_path: str) -> Path:
    root = Path(workspace_root).resolve()
    candidate = (root / file_path).resolve() if not os.path.isabs(file_path) else Path(file_path).resolve()
    candidate.relative_to(root)
    return candidate


def validate_code(code: str) -> Dict[str, Any]:
    try:
        ast.parse(code)
        return {"ok": True, "message": "Syntax OK"}
    except SyntaxError as e:
        return {"ok": False, "error": str(e), "lineno": e.lineno, "offset": e.offset}


def save_code(workspace_root: str, relative_path: str, code: str) -> Dict[str, Any]:
    if not (code or "").strip():
        return {"ok": False, "error": "code is empty; pass the full script contents"}
    try:
        p = _resolve_under_workspace(workspace_root, relative_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(code, encoding="utf-8")
        return {"ok": True, "path": str(p)}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except OSError as e:
        return {"ok": False, "error": str(e)}


def _execution_path_prelude(workspace_root: str) -> str:
    """So agents can `import clean_data` when the file lives at data/clean_data.py."""
    root = str(Path(workspace_root).resolve())
    return f"""import os, sys
_ws = {root!r}
# This runs as plain Python — agent tool names (tool_read_data, tool_execute_code, …) do NOT exist here.
# Load CSVs with pandas: pd.read_csv("train.csv"), pd.read_csv("test.csv").
if _ws not in sys.path:
    sys.path.insert(0, _ws)
for _sub in ("data", "src", "scripts", "notebooks", "lib"):
    _p = os.path.join(_ws, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
"""


def execute_code(workspace_root: str, code: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Write code to a temp file under workspace and run with the same Python interpreter.
    Workspace root and common subdirs (data/, src/, …) are on sys.path so nested modules import.
    """
    if not (code or "").strip():
        return {"ok": False, "error": "code is empty; pass a non-empty Python script"}
    syn = validate_code(code)
    if not syn.get("ok"):
        return {
            "ok": False,
            "returncode": 1,
            "stdout": "",
            "stderr": f"Syntax error (not run): {syn.get('error', syn)}",
        }
    root = Path(workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    full_source = _execution_path_prelude(str(root)) + "\n" + code
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
            dir=root,
        ) as f:
            f.write(full_source)
            script_path = f.name
        proc = subprocess.run(
            [sys.executable, script_path],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-20000:] if proc.stdout else "",
            "stderr": proc.stderr[-20000:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        try:
            if "script_path" in locals() and os.path.isfile(script_path):
                os.unlink(script_path)
        except OSError:
            pass
