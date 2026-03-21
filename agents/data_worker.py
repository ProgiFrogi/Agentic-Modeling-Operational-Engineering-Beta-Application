from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.llm_factory import get_chat_llm
from agents.prompts import DATA_WORKER_SYSTEM
from agents.tool_loop import invoke_with_tools
from tools.agent_tools import build_data_worker_tools


def run_data_worker(state: dict) -> dict:
    ws = state["workspace_dir"]
    tools = build_data_worker_tools(ws)
    model = get_chat_llm()
    user = (
        f"Competition: {state.get('competition_ref')}\n"
        f"Planner instruction:\n{state.get('planner_instruction', '')}\n"
        f"Prior analysis:\n{state.get('data_analysis', '')}\n"
    )
    text = invoke_with_tools(
        model,
        tools,
        DATA_WORKER_SYSTEM,
        user,
        max_rounds=14,
        agent_label="data_worker",
        workspace_dir=ws,
    )
    return {
        "data_work_summary": text,
        "messages": [AIMessage(content=f"[DataWorker]\n{text}")],
    }
