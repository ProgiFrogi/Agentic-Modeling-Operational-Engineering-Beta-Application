from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph

from agents.coder import run_coder
from agents.data_analytic import run_data_analytic
from agents.data_worker import run_data_worker
from agents.performance_assessor import run_performance_assessor
from agents.planner import run_planner
from config.settings import get_settings
from tools.data_tools import ensure_workspace_kaggle_csv_aliases
from tools.kaggle_utils import submit_solution
from workflows.planner_routing import planner_route_after_decision
from workflows.state import AgentState

_workflow_log = logging.getLogger("agentic.workflow")


def _describe_state(st: dict[str, Any]) -> None:
    ws = st.get("workspace_dir") or "."
    sub = os.path.join(ws, "submission.csv")
    _workflow_log.info(
        "state iter=%s next_step=%s msgs=%s submission_exists=%s",
        st.get("iteration"),
        st.get("next_step"),
        len(st.get("messages") or []),
        os.path.isfile(sub),
    )
    if st.get("planner_instruction"):
        _workflow_log.info(
            "  planner_instruction: %s",
            str(st.get("planner_instruction")).replace("\n", " ")[:400],
        )
    if st.get("submit_result"):
        _workflow_log.info("  submit_result: %s", str(st.get("submit_result"))[:500])
    if st.get("assessor_feedback"):
        _workflow_log.info(
            "  assessor_feedback: %s",
            str(st.get("assessor_feedback")).replace("\n", " ")[:400],
        )


def _submission_full_path(state: AgentState) -> str:
    ws = state.get("workspace_dir") or "."
    rel = state.get("submission_path") or "submission.csv"
    return rel if os.path.isabs(rel) else os.path.join(ws, rel)


def _route_after_planner(state: AgentState) -> Literal[
    "data_analytic", "data_worker", "coder", "submit_action", "assessor", "end"
]:
    s = get_settings()
    target, _ = planner_route_after_decision(state)
    it = int(state.get("iteration", 0))
    if (
        it > s.max_workflow_iterations
        and target == "end"
        and state.get("next_step") != "end"
    ):
        _workflow_log.warning(
            "stopping: iteration %s exceeds MAX_WORKFLOW_ITERATIONS=%s",
            state.get("iteration"),
            s.max_workflow_iterations,
        )
    return target


def _submit_action(state: AgentState) -> dict:
    ws = state.get("workspace_dir") or "."
    rel = state.get("submission_path") or "submission.csv"
    full = rel if os.path.isabs(rel) else os.path.join(ws, rel)
    comp = state.get("competition_ref") or get_settings().competition_ref
    slog = logging.getLogger("agentic.submit")
    if not os.path.isfile(full):
        slog.warning("skip Kaggle submit: file missing %s", full)
        result = {
            "ok": False,
            "error": f"submission file missing: {full} (Coder must write submission.csv in workspace)",
        }
    else:
        slog.info("submitting to Kaggle competition=%s file=%s", comp, full)
        result = submit_solution(full, comp, message="multi-agent submission", quiet=True)
        slog.info("submit result: %s", result)
    return {
        "submit_result": str(result),
        "messages": [AIMessage(content=f"[Submit]{result}")],
    }


def build_workflow_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", run_planner)
    g.add_node("data_analytic", run_data_analytic)
    g.add_node("data_worker", run_data_worker)
    g.add_node("coder", run_coder)
    g.add_node("submit_action", _submit_action)
    g.add_node("assessor", run_performance_assessor)

    g.set_entry_point("planner")
    g.add_conditional_edges(
        "planner",
        _route_after_planner,
        {
            "end": END,
            "data_analytic": "data_analytic",
            "data_worker": "data_worker",
            "coder": "coder",
            "submit_action": "submit_action",
            "assessor": "assessor",
        },
    )
    g.add_edge("data_analytic", "planner")
    g.add_edge("data_worker", "planner")
    g.add_edge("coder", "planner")
    g.add_edge("submit_action", "assessor")
    g.add_edge("assessor", "planner")
    return g.compile()


def run_kaggle_workflow(
    competition_ref: str | None = None,
    workspace_dir: str | None = None,
    run_id: str | None = None,
    verbose: bool = True,
) -> AgentState:
    s = get_settings()
    comp = competition_ref or s.competition_ref
    ws = workspace_dir or str(Path(s.workspace_root) / comp.split("/")[-1])
    os.makedirs(ws, exist_ok=True)
    linked = ensure_workspace_kaggle_csv_aliases(ws)
    if linked:
        _workflow_log.info("created CSV path aliases for confused agent paths: %s", linked)
    graph = build_workflow_graph()
    init: AgentState = {
        "messages": [],
        "competition_ref": comp,
        "workspace_dir": ws,
        "iteration": 0,
        "plan": "",
        "run_id": run_id or str(uuid.uuid4()),
    }
    config = {"recursion_limit": 80}
    if not verbose:
        return graph.invoke(init, config)  # type: ignore[return-value]

    _workflow_log.info("start competition=%s workspace=%s run_id=%s", comp, ws, init["run_id"])
    print(f"[workflow] competition={comp} workspace={ws} run_id={init['run_id']}", file=sys.stderr, flush=True)
    last: Optional[AgentState] = None
    for state in graph.stream(init, config, stream_mode="values"):
        last = state  # type: ignore[assignment]
        st = dict(state) if state is not None else {}
        _describe_state(st)
        nmsg = len(st.get("messages") or [])
        line = (
            f"[workflow] iter={st.get('iteration')} next={st.get('next_step')!r} "
            f"messages={nmsg}"
        )
        print(line, file=sys.stderr, flush=True)
    _workflow_log.info("done")
    print("[workflow] done.", file=sys.stderr, flush=True)
    if last is not None:
        return last  # type: ignore[return-value]
    return graph.invoke(init, config)  # type: ignore[return-value]
