"""Small ReAct-style loop for tool-calling agents."""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

_log = logging.getLogger("agentic.tools")


def _trunc(s: str, n: int = 600) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def invoke_with_tools(
    model: ChatOpenAI,
    tools: Sequence[BaseTool],
    system: str,
    user: str,
    max_rounds: int = 8,
    agent_label: str = "agent",
    workspace_dir: Optional[str] = None,
) -> str:
    messages: List = [SystemMessage(content=system), HumanMessage(content=user)]
    bound = model.bind_tools(list(tools))
    tools_were_used = False
    nudge_prose_without_tools = 0
    data_worker_executed = False
    data_worker_execute_nudges = 0
    coder_submission_nudges = 0
    for round_i in range(max_rounds):
        resp = bound.invoke(messages)
        messages.append(resp)
        if not isinstance(resp, AIMessage) or not resp.tool_calls:
            text = (resp.content or "").strip()
            if (
                agent_label == "data_worker"
                and not tools_were_used
                and nudge_prose_without_tools < 2
            ):
                nudge_prose_without_tools += 1
                messages.append(
                    HumanMessage(
                        content="You must call tools in this turn: tool_save_code (with relative_path) "
                        "then tool_execute_code. Do not answer with prose only."
                    )
                )
                continue
            if (
                agent_label == "data_worker"
                and tools_were_used
                and not data_worker_executed
                and data_worker_execute_nudges < 6
            ):
                data_worker_execute_nudges += 1
                messages.append(
                    HumanMessage(
                        content="You still have not successfully run tool_execute_code. "
                        "Save a .py file with tool_save_code, then run it with tool_execute_code "
                        "(or run a one-liner script with pd.read_csv). validate_code is not enough."
                    )
                )
                continue
            if agent_label == "coder" and workspace_dir:
                sub = os.path.join(workspace_dir, "submission.csv")
                if not os.path.isfile(sub) and coder_submission_nudges < 8:
                    coder_submission_nudges += 1
                    messages.append(
                        HumanMessage(
                            content="submission.csv is still missing on disk. "
                            "Call tool_execute_code with a complete script using pd.read_csv only "
                            "(no tool_* names inside the script). Write submission.csv with columns "
                            "index,prediction matching sample_submition.csv. Then tool_check_file('submission.csv')."
                        )
                    )
                    continue
            _log.info("[%s] round=%s final_reply=%s", agent_label, round_i, _trunc(text, 400))
            return text
        tools_were_used = True
        _log.info("[%s] round=%s tool_calls=%s", agent_label, round_i, len(resp.tool_calls))
        for tc in resp.tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            tool_fn = next((t for t in tools if t.name == name), None)
            if tool_fn is None:
                out = f"unknown tool: {name}"
            else:
                try:
                    out = tool_fn.invoke(args)
                except Exception as e:
                    out = f"tool error: {e}"
            _log.info(
                "[%s] tool=%s args=%s -> %s",
                agent_label,
                name,
                _trunc(str(args), 300),
                _trunc(str(out), 400),
            )
            messages.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
            if agent_label == "data_worker" and name == "tool_execute_code":
                out_s = str(out)
                if "'ok': True" in out_s or '"ok": True' in out_s or "returncode': 0" in out_s:
                    data_worker_executed = True
            if agent_label == "coder" and name == "tool_execute_code":
                out_s = str(out)
                if (
                    "tool_read_data" in out_s
                    or "tool_inspect_data" in out_s
                    or ("tool_" in out_s and "not defined" in out_s)
                    or "NameError" in out_s
                ):
                    messages.append(
                        HumanMessage(
                            content="execute_code runs plain Python: tool_* functions do not exist there. "
                            "Rewrite using pd.read_csv('train.csv') and pd.read_csv('test.csv') only."
                        )
                    )
    _log.warning("[%s] max tool rounds exceeded (%s)", agent_label, max_rounds)
    return "max tool rounds exceeded"
