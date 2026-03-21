"""Single source for post-planner routing: LangGraph target + optional next_step override."""

from __future__ import annotations

import os
from typing import Any, Dict, Literal, Tuple

from config.settings import get_settings

GraphTarget = Literal[
    "data_analytic", "data_worker", "coder", "submit_action", "assessor", "end"
]


def planner_route_after_decision(state: Dict[str, Any]) -> Tuple[GraphTarget, Dict[str, Any]]:
    """
    After the planner LLM updates state, decide which graph node runs next.

    Returns (graph_node_key, state_patch). state_patch may set next_step when we override
    the planner (e.g. force coder) so logs and specialist context stay consistent.
    """
    s = get_settings()
    patch: Dict[str, Any] = {}

    if state.get("next_step") == "end":
        return "end", patch

    if int(state.get("iteration", 0)) > s.max_workflow_iterations:
        return "end", patch

    step = state.get("next_step") or "end"
    ws = state.get("workspace_dir") or "."
    rel = state.get("submission_path") or "submission.csv"
    sub_path = rel if os.path.isabs(rel) else os.path.join(ws, rel)
    sub_missing = not os.path.isfile(sub_path)
    it = int(state.get("iteration", 0))

    if (
        sub_missing
        and it >= s.force_coder_min_planner_iteration
        and step in ("data_analytic", "data_worker")
    ):
        patch["next_step"] = "coder"
        return "coder", patch
    
    # Skip data_worker entirely after EDA - go straight to coder
    if step == "data_worker" and sub_missing:
        patch["next_step"] = "coder"
        return "coder", patch

    if step == "submit":
        if sub_missing:
            patch["next_step"] = "coder"
            return "coder", patch
        return "submit_action", patch

    if step == "assessor":
        if sub_missing:
            patch["next_step"] = "coder"
            return "coder", patch
        return "assessor", patch

    if step in ("data_analytic", "data_worker", "coder"):
        return step, patch  # type: ignore[return-value]

    return "end", patch
