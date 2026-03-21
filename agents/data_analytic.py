from __future__ import annotations

from langchain_core.messages import AIMessage

from agents.llm_factory import get_chat_llm
from agents.prompts import DATA_ANALYTIC_SYSTEM
from agents.tool_loop import invoke_with_tools
from tools.agent_tools import build_data_analytic_tools


def run_data_analytic(state: dict) -> dict:
    ws = state["workspace_dir"]
    tools = build_data_analytic_tools(ws)
    model = get_chat_llm()
    user = (
        f"Competition: {state.get('competition_ref')}\n"
        f"Planner instruction:\n{state.get('planner_instruction', '')}\n"
        f"Inspect train.csv and test.csv under the workspace. Use tools.\n"
    )
    text = invoke_with_tools(
        model, tools, DATA_ANALYTIC_SYSTEM, user, max_rounds=8, agent_label="data_analytic"
    )
    return {
        "data_analysis": text,
        "messages": [AIMessage(content=f"[DataAnalytic]\n{text}")],
    }
