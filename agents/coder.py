from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.llm_factory import get_chat_llm
from agents.prompts import CODER_SYSTEM
from agents.tool_loop import invoke_with_tools
from tools.agent_tools import build_coder_tools


def run_coder(state: dict) -> dict:
    ws = state["workspace_dir"]
    tools = build_coder_tools(ws)
    model = get_chat_llm()
    user = (
        f"Competition: {state.get('competition_ref')}\n"
        f"Planner instruction:\n{state.get('planner_instruction', '')}\n"
        f"Data analysis:\n{state.get('data_analysis', '')}\n"
        f"Data work:\n{state.get('data_work_summary', '')}\n"
        "Deliverable: submission.csv in workspace root. Sample file is usually sample_submition.csv "
        "with columns index,prediction — match that format exactly.\n"
    )
    text = invoke_with_tools(
        model,
        tools,
        CODER_SYSTEM,
        user,
        max_rounds=18,
        agent_label="coder",
        workspace_dir=ws,
    )
    return {
        "code_summary": text,
        "submission_path": "submission.csv",
        "messages": [AIMessage(content=f"[Coder]\n{text}")],
    }
