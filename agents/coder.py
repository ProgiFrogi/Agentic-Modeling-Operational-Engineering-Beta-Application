""" Main model for code """


import os
import re
from typing import Dict, Any, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from tools import extract_code, check_syntax
from tools import execute_with_saving as execute
from dotenv import load_dotenv
from utils import logger

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
    data_dir: Optional[str]

llm = ChatOllama(
    model="qwen2.5-coder:14b-instruct-q4_K_M",
    temperature=0,
)


def generate_initial_code(state: CodingAgentState) -> Dict[str, Any]:
    prompt = f"""
    Write Python code for the following task:\n{state['task']}\nImportant: When generating code, ensure:
    - Use proper spacing: "import pandas as pd" not "import pandasaspd"
    - Add spaces after commas: "func(arg1, arg2)" not "func(arg1,arg2)"
    - Put each statement on a new line
    - Use proper indentation (4 spaces per level)
    - Ensure the code runs without errors and outputs meaningful results.
    - If using sklearn, use sparse_output=False instead of sparse=False for OneHotEncoder (newer sklearn versions)
    - Use .ravel() when assigning imputed values to avoid dimension issues
    - Don't use matplotlib or seaborn for visualizations
    
    1. Memory efficiency is KEY - work with large datasets (36k rows, 15 columns)
    2. DO NOT create explosion of features - avoid OneHotEncoder on high-cardinality columns
    3. For columns with > 50 unique values, use:
       - Label Encoding, or
       - Frequency encoding, or
       - Keep as-is (if meaningful for model like text)
    4. Always print shapes and memory usage after operations
    5. Handle missing values simply:
       - Numerical: fill with median/mean
       - Categorical: fill with "Unknown" or mode
    6. For datetime: extract useful features (year, month, day, dayofweek)
    7. Use StandardScaler for numerical features
    8. Save processed files with .to_csv(index=False)
    9. Print progress messages
    10. **TEST DATA DOES NOT HAVE 'target' COLUMN** - This is the most important rule!
       - Always exclude 'target' from numerical_cols when processing test data
       - Use: numerical_cols_train = [c for c in numerical_cols if c != 'target']
       - For test: use the same columns as train WITHOUT 'target'
    REMEMBER - in test.csv not 'target'
    Output only the code, no explanations.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    code = extract_code(response.content)
    logger.info(f"Coder Request: {prompt}")
    logger.info(f"Coder Response: {code}")
    if not state['task'] or (state['task'] == "" and state.get("done", False)):
        logger.info("[Wrapper] No task or already done, skipping coder")
        return {"done": True}

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
    data_dir = state.get("data_dir", None)
    task_name = re.sub(r'[^\w\-_\. ]', '_', state['task'][:50])
    output_dir = f"execution_results/{task_name}"
    success, output, result_dir = execute(code, data_dir=data_dir, output_dir=output_dir)



    if success:
        logger.info(f"Code executed successfully! Results saved to {result_dir}")
        return {
            "execution_output": output,
            "execution_error": None,
            "done": True,
            "final_code": code
        }
    else:
        logger.info(f"Code executed unsuccessfully: {output}")
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
    logger.info(f"Coder Request: {prompt}")
    logger.info(f"Coder Response: {new_code}")
    return {
        "current_code": new_code,
        "attempts": state["attempts"] + 1,
        "syntax_error": None,
        "execution_error": None,
        "execution_output": None,
        "done": False
    }

def after_check(state: CodingAgentState) -> str:
    if state["syntax_error"] and state["attempts"] < state["max_attempts"]:
        return "fix"
    elif not state["syntax_error"]:
        return "execute"
    return "fail"

def after_execute(state: CodingAgentState) -> str:
    if state["execution_error"] is None:
        return "success"
    elif state["attempts"] < state["max_attempts"]:
        return "fix"
    else:
        return "fail"

def run_coder(task: str, max_attempts: int = 3, data_dir: str | None = None) -> Dict[str, Any]:
    graph = StateGraph(CodingAgentState)
    graph.add_node("generate", generate_initial_code)
    graph.add_node("check_syntax", check_syntax)
    graph.add_node("execute", execute_code)
    graph.add_node("fix", fix_code)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "check_syntax")
    graph.add_conditional_edges("check_syntax", after_check, {"execute": "execute", "fix": "fix", "fail": END})
    graph.add_conditional_edges("execute", after_execute, {"success": END, "fix": "fix", "fail": END})
    graph.add_edge("fix", "check_syntax")

    app = graph.compile()

    initial_state = {
        "task": task,
        "current_code": "",
        "done": False,
        "syntax_error": None,
        "execution_error": None,
        "execution_output": None,
        "attempts": 0,
        "max_attempts": max_attempts,
        "final_code": None,
        "data_dir": data_dir,
    }

    result = app.invoke(initial_state)

    print("="*50)
    print("Final code:")
    print(result.get("final_code") or result.get("current_code", "No code"))
    print("Execution output:", result.get("execution_output"))
    print("Execution error:", result.get("execution_error"))
    return result