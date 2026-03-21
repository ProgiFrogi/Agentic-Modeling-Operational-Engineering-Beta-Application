"""LangChain Tool wrappers bound to a workspace for agent nodes."""

from __future__ import annotations

import uuid
from typing import List, Optional

from langchain_core.tools import tool

from tools.code_tools import execute_code, save_code, validate_code
from tools.data_tools import check_file, inspect_data, list_workspace_files, read_data
from tools.rag_tools import retrieve_code_as_json


def build_agent_toolkit(workspace_root: str):
    @tool
    def tool_check_file(file_path: str) -> str:
        """Return JSON-ish dict whether file exists under workspace. Use train.csv / test.csv at workspace root (not data/train.csv)."""
        return str(check_file(workspace_root, file_path))

    @tool
    def tool_list_workspace(glob_pattern: str = "*", max_files: int = 500) -> str:
        """List files under workspace. Do not set max_files below ~20 or you may hide train.csv/test.csv. Default 500 is safe."""
        return str(
            list_workspace_files(
                workspace_root,
                glob_pattern=glob_pattern,
                max_files=int(max_files),
            )
        )

    @tool
    def tool_read_data(file_path: str, nrows: Optional[int] = None) -> str:
        """Preview CSV (columns, dtypes, head). nrows defaults to 200 if omitted or null; use inspect_data for full-file stats."""
        nr = 200 if nrows is None else int(nrows)
        return str(read_data(workspace_root, file_path, nrows=nr))

    @tool
    def tool_inspect_data(file_path: str) -> str:
        """Full shape, nulls, numeric summary. Use train.csv at workspace root, not data/train.csv."""
        return str(inspect_data(workspace_root, file_path))

    @tool
    def tool_validate_code(code: str) -> str:
        """Parse Python code for syntax errors."""
        return str(validate_code(code))

    @tool
    def tool_execute_code(code: str) -> str:
        """Run Python in workspace (subprocess). Use pd.read_csv — never call tool_read_data inside this code."""
        from config.settings import get_settings

        return str(execute_code(workspace_root, code, timeout=get_settings().code_execution_timeout))

    @tool
    def tool_save_code(code: str, relative_path: Optional[str] = None) -> str:
        """Write a file under workspace. Prefer an explicit relative_path (e.g. data/clean_data.py); if omitted, a unique scripts/_agent_*.py path is used."""
        rp = (relative_path or "").strip() or f"scripts/_agent_{uuid.uuid4().hex[:10]}.py"
        return str(save_code(workspace_root, rp, code))

    @tool
    def tool_retrieve_code(query: str, n_results: int = 3) -> str:
        """Search indexed Kaggle notebook code chunks (RAG)."""
        return retrieve_code_as_json(query, n_results=int(n_results))

    return [
        tool_check_file,
        tool_list_workspace,
        tool_read_data,
        tool_inspect_data,
        tool_validate_code,
        tool_execute_code,
        tool_save_code,
        tool_retrieve_code,
    ]


def build_data_analytic_tools(workspace_root: str):
    t = build_agent_toolkit(workspace_root)
    return [t[0], t[1], t[2], t[3]]  # check, list, read, inspect


def build_data_worker_tools(workspace_root: str):
    t = build_agent_toolkit(workspace_root)
    # read + inspect: planner often asks for inspect_data; avoid fake Python modules
    return [t[0], t[1], t[2], t[3], t[4], t[5], t[6]]  # check, list, read, inspect, validate, execute, save


def build_coder_tools(workspace_root: str):
    return build_agent_toolkit(workspace_root)
