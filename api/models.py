from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CompetitionStartRequest(BaseModel):
    competition_ref: Optional[str] = Field(
        default=None,
        description="Kaggle competition slug (defaults to COMPETITION_REF from env)",
    )
    workspace_dir: Optional[str] = Field(
        default=None,
        description="Absolute or relative workspace path for data and outputs",
    )


class CompetitionStartResponse(BaseModel):
    run_id: str
    status: str = "started"


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WsEvent(BaseModel):
    type: str
    payload: Dict[str, Any] = Field(default_factory=dict)


def state_to_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages") or []
    contents: List[str] = []
    for m in msgs:
        c = getattr(m, "content", None)
        if c:
            contents.append(str(c)[:4000])
    return {
        "competition_ref": state.get("competition_ref"),
        "workspace_dir": state.get("workspace_dir"),
        "iteration": state.get("iteration"),
        "next_step": state.get("next_step"),
        "plan": state.get("plan"),
        "data_analysis": state.get("data_analysis"),
        "data_work_summary": state.get("data_work_summary"),
        "code_summary": state.get("code_summary"),
        "submission_path": state.get("submission_path"),
        "assessor_feedback": state.get("assessor_feedback"),
        "last_public_score": state.get("last_public_score"),
        "submit_result": state.get("submit_result"),
        "messages_tail": contents[-30:],
    }
