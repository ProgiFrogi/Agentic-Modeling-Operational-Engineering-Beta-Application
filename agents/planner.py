from __future__ import annotations

import logging
import os
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.llm_factory import get_planner_llm
from agents.prompts import PLANNER_SYSTEM
from workflows.planner_routing import planner_route_after_decision

_log = logging.getLogger("agentic.planner")


class PlannerDecision(BaseModel):
    next_step: Literal["data_analytic", "data_worker", "coder", "submit", "assessor", "end"] = Field(
        description="Which specialist runs next"
    )
    instruction: str = Field(description="Concrete task for that specialist")
    plan_delta: str = Field(description="Short plan note")


def run_planner(state: dict) -> dict:
    _log.info(
        "planner enter iteration=%s (about to call LLM)",
        state.get("iteration", 0),
    )
    llm = get_planner_llm().with_structured_output(PlannerDecision)
    ws = state.get("workspace_dir") or "."
    sub_fp = os.path.join(ws, "submission.csv")
    sub_ok = os.path.isfile(sub_fp)
    sub_human = "YES" if sub_ok else "NO"
    ctx = (
        f"Competition: {state.get('competition_ref')}\n"
        f"Workspace: {ws}\n"
        f"=== SUBMISSION FILE (ground truth; obey this) ===\n"
        f"submission.csv present on disk: {sub_human}\n"
        f"If NO: you MUST choose next_step=coder (or data_* only for quick prep), NEVER submit or assessor.\n"
        f"sample_submition.csv / sample_submission.csv is ONLY the row-id template — NOT submission.csv.\n"
        f"=== end submission block ===\n"
        f"Planner step (1-based pass): {int(state.get('iteration', 0)) + 1}\n"
        f"Current plan:\n{state.get('plan', '')}\n"
        f"Data analysis notes:\n{state.get('data_analysis', '')}\n"
        f"Data work notes:\n{state.get('data_work_summary', '')}\n"
        f"Code notes:\n{state.get('code_summary', '')}\n"
        f"Assessor feedback:\n{state.get('assessor_feedback', '')}\n"
        f"Last public score info:\n{state.get('last_public_score', '')}\n"
    )
    decision: PlannerDecision = llm.invoke(
        [SystemMessage(content=PLANNER_SYSTEM), HumanMessage(content=ctx)]
    )
    _log.info(
        "planner decision next_step=%s instruction=%s",
        decision.next_step,
        str(decision.instruction).replace("\n", " ")[:300],
    )
    plan = (state.get("plan") or "").strip()
    if plan:
        plan += "\n"
    plan += decision.plan_delta
    new_iter = int(state.get("iteration", 0)) + 1
    merged = {**state, "next_step": decision.next_step, "iteration": new_iter}
    _, route_patch = planner_route_after_decision(merged)
    effective_step = route_patch.get("next_step", decision.next_step)
    instruction_for_specialist = decision.instruction
    if route_patch.get("next_step") == "coder" and decision.next_step != "coder":
        _log.warning(
            "overriding planner next_step=%s -> coder (no submission.csv yet or too much EDA without submit file)",
            decision.next_step,
        )
        if decision.next_step in ("submit", "assessor"):
            instruction_for_specialist = (
                "submission.csv is missing in the workspace root. Train a baseline on train.csv, predict "
                "test.csv, and write submission.csv (same id column as the sample template). "
                "Ignore any claim that submission.csv already exists until the file is on disk."
            )
        else:
            instruction_for_specialist = (
                f"{decision.instruction}\n\n"
                "Priority: produce submission.csv in the workspace root next (baseline model is fine)."
            )
    msg = AIMessage(
        content=(
            f"[Planner] next={effective_step}\n{instruction_for_specialist}\n{decision.plan_delta}"
        )
    )
    out = {
        "next_step": effective_step,
        "plan": plan,
        "planner_instruction": instruction_for_specialist,
        "messages": [msg],
        "iteration": new_iter,
    }
    return out
