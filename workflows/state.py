from __future__ import annotations

from typing import Annotated, List, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Shared LangGraph state for the Kaggle multi-agent workflow."""

    messages: Annotated[List[AnyMessage], add_messages]
    competition_ref: str
    workspace_dir: str
    iteration: int
    next_step: str
    planner_instruction: str
    plan: str
    data_analysis: str
    data_work_summary: str
    code_summary: str
    submission_path: str
    assessor_feedback: str
    last_public_score: str
    submit_result: str
    run_id: str
