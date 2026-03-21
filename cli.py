#!/usr/bin/env python3
"""CLI entrypoint for the multi-agent Kaggle workflow."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from config.settings import get_settings
from tools.kaggle_utils import setup_competition
from tools.workspace_cleanup import clean_agent_artifacts, fork_workspace_with_data_symlinks
from workflows.kaggle_workflow import run_kaggle_workflow


def _configure_cli_logging(quiet: bool, workflow_verbose: bool) -> None:
    if quiet or not workflow_verbose:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s", force=True)
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
        force=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Run LangGraph Kaggle multi-agent workflow")
    p.add_argument("--competition", default=None, help="Kaggle competition slug")
    p.add_argument("--workspace", default=None, help="Workspace directory for data/outputs")
    p.add_argument(
        "--download-data",
        action="store_true",
        help="Download competition files into workspace before running agents",
    )
    p.add_argument(
        "--download-only",
        action="store_true",
        help="Only download data via Kaggle API, then exit (no LLM / no agent workflow)",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Less output: no per-step workflow stream on stderr, agent logs at WARNING",
    )
    p.add_argument(
        "--clean-agent-artifacts",
        action="store_true",
        help="Before running: remove scripts/_agent_*.py, tmp*.py, __pycache__ under workspace (avoids stale imports).",
    )
    p.add_argument(
        "--fork-workspace",
        action="store_true",
        help="Run in a new timestamped folder with symlinks to CSVs from the usual competition workspace (clean run, same data).",
    )
    args = p.parse_args()
    s = get_settings()
    _configure_cli_logging(quiet=args.quiet, workflow_verbose=s.workflow_verbose)
    comp = args.competition or s.competition_ref
    if args.workspace:
        parent_ws = args.workspace
    elif s.workspace_path:
        parent_ws = s.workspace_path
    else:
        parent_ws = str(Path(s.workspace_root) / comp.split("/")[-1])

    if args.fork_workspace:
        os.makedirs(parent_ws, exist_ok=True)
        try:
            ws = fork_workspace_with_data_symlinks(parent_ws)
        except FileNotFoundError:
            print(
                "fork-workspace: base workspace has no data. "
                "Run once with --download-data on the default workspace, or pass --workspace pointing at a folder that already contains train.csv."
            )
            raise SystemExit(2)
        print(f"Using forked workspace: {ws}")
    else:
        ws = parent_ws

    os.makedirs(ws, exist_ok=True)
    if args.clean_agent_artifacts:
        r = clean_agent_artifacts(ws)
        print(f"clean-agent-artifacts: removed {r.get('removed_count', 0)} path(s)")
    if args.download_data or args.download_only:
        print(setup_competition(comp, ws))
    if args.download_only:
        return
    # Подробный поток графа по умолчанию (WORKFLOW_VERBOSE=1); выключить: --quiet или WORKFLOW_VERBOSE=0
    verbose = (not args.quiet) and s.workflow_verbose
    log = logging.getLogger("agentic.cli")
    log.info("competition=%s workspace=%s max_iterations=%s", comp, ws, s.max_workflow_iterations)
    result = run_kaggle_workflow(competition_ref=comp, workspace_dir=ws, verbose=verbose)
    sub = os.path.join(ws, "submission.csv")
    log.info(
        "finished iteration=%s next_step=%s messages=%s submission.csv_exists=%s",
        result.get("iteration"),
        result.get("next_step"),
        len(result.get("messages") or []),
        os.path.isfile(sub),
    )
    print(
        "Finished. iteration=",
        result.get("iteration"),
        "next_step=",
        result.get("next_step"),
        "messages=",
        len(result.get("messages") or []),
        "submission.csv_exists=",
        os.path.isfile(sub),
    )


if __name__ == "__main__":
    main()
