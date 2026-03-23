import os
from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tools import extract_code, execute_in_docker, check_syntax
from dotenv import load_dotenv

load_dotenv()

class CodingAgentState(TypedDict):
    task: str
    current_code: str
    syntax_error: Optional[str]
    execution_error: Optional[str]
    execution_output: Optional[str]
    attempts: int
    max_attempts: int
    done: bool
    final_code: Optional[str]

llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
)


def generate_initial_code(state: CodingAgentState) -> Dict[str, Any]:
    prompt = f"Write Python code for the following task:\n{state['task']}\nOutput only the code, no explanations."
    response = llm.invoke([HumanMessage(content=prompt)])
    code = extract_code(response.content)
    return {
        "current_code": code,
        "attempts": 1,
        "done": False,
        "syntax_error": None,
        "execution_error": None,
        "execution_output": None,
        "final_code": None
    }

def execute_code(state: CodingAgentState) -> Dict[str, Any]:
    code = state["current_code"]
    success, output = execute_in_docker(code)
    if success:
        return {
            "execution_output": output,
            "execution_error": None,
            "done": True,
            "final_code": code
        }
    else:
        return {
            "execution_error": output,
            "execution_output": None,
            "done": True
        }

def fix_code(state: CodingAgentState) -> Dict[str, Any]:
    error = state.get("syntax_error") or state.get("execution_error")
    prompt = f"""The following code had an error:
{state['current_code']}
Error: {error}
Please provide correct code. Output only the code, no explanations.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    new_code = extract_code(response.content)
    return {
        "current_code": new_code,
        "attempts": state["attempts"] + 1,
        "syntax_error": None,
        "execution_error": None,
        "execution_output": None,
        "done": False
    }

def after_check(state: CodingAgentState) -> str:
    if state["syntax_error"]:
        return "fix"
    return "execute"

def after_execute(state: CodingAgentState) -> str:
    if state["execution_error"] is None:
        return "success"
    elif state["attempts"] < state["max_attempts"]:
        return "fix"
    else:
        return "fail"

# Строим граф
graph = StateGraph(CodingAgentState)
graph.add_node("generate", generate_initial_code)
graph.add_node("check_syntax", check_syntax)
graph.add_node("execute", execute_code)
graph.add_node("fix", fix_code)

graph.set_entry_point("generate")
graph.add_edge("generate", "check_syntax")
graph.add_conditional_edges("check_syntax", after_check, {"execute": "execute", "fix": "fix"})
graph.add_conditional_edges("execute", after_execute, {"success": END, "fix": "fix", "fail": END})
graph.add_edge("fix", "check_syntax")

app = graph.compile()

initial_state = {
    "task": "Write a function that sorts an array and returns it.",
    "current_code": "",
    "done": False,
    "syntax_error": None,
    "execution_error": None,
    "execution_output": None,
    "attempts": 0,
    "max_attempts": 3,
    "final_code": None
}

result = app.invoke(initial_state)

print("="*50)
print("Final code:")
print(result.get("final_code") or result.get("current_code", "No code"))
print("Execution output:", result.get("execution_output"))
print("Execution error:", result.get("execution_error"))