from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agents.llm_factory import get_chat_llm
from agents.prompts import ASSESSOR_SYSTEM
from tools.kaggle_utils import check_submission_status

_log = logging.getLogger("agentic.assessor")


class AssessorReport(BaseModel):
    feedback: str = Field(description="Feedback for Planner")
    suggested_next_focus: str = Field(description="One primary focus area")


def run_performance_assessor(state: dict) -> dict:
    comp = state["competition_ref"]
    _log.info("assessor fetching submissions for %s", comp)
    subs = check_submission_status(comp, limit=5, quiet=True)
    _log.info("assessor got %s submission row(s)", len(subs))
    subs_str = json.dumps(subs, ensure_ascii=False, default=str)[:12000]
    llm = get_chat_llm().with_structured_output(AssessorReport)
    rep: AssessorReport = llm.invoke(
        [
            SystemMessage(content=ASSESSOR_SYSTEM),
            HumanMessage(
                content=f"Recent submissions JSON:\n{subs_str}\n"
                f"Code/coder notes:\n{state.get('code_summary', '')}\n"
                f"Last submit attempt:\n{state.get('submit_result', '')}\n"
            ),
        ]
    )
    last_score = ""
    if subs:
        last_score = str(subs[0].get("public_score") or subs[0].get("status"))
    return {
        "assessor_feedback": rep.feedback,
        "last_public_score": last_score,
        "messages": [
            AIMessage(
                content=f"[Assessor] {rep.feedback}\nNext focus: {rep.suggested_next_focus}"
            )
        ],
    }
