""" Analyze data and give instructions to data_worker """


import os
from typing import Dict, Any, TypedDict, List, Optional

import json
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utils import data_utils, logger
from dotenv import load_dotenv
from agents.prompts import data_analytic_prompt

from agents.coder import run_coder

load_dotenv()

class DataWorkState(TypedDict):
    task: str
    df: pd.DataFrame
    data_dir: Optional[str]
    path_to_comp_desc: str
    name_of_file: str

    current_plan: str
    analytic_attempts: int
    analytic_max_attempts: int
    satisfy_rate: float

    worker_attempts: int
    worker_max_attempts: int

    previous_results: List[Dict[str, Any]]
    done: bool


llm = ChatOpenAI(
    model="stepfun/step-3.5-flash:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
    openai_api_base="https://openrouter.ai/api/v1",
)

# Промпт состоит из: описания датасета, описания колонок, что сейчас есть, информации о NaN
def analytic_initial_step(state: DataWorkState) -> Dict[str, Any]:
    request_data = data_utils.get_initial_data_form_df(state["df"], state["path_to_comp_desc"], state["name_of_file"])
    request_data["history"] = "It's your first try"
    prompt = (data_analytic_prompt.initial_prompt
              .format(**request_data))
    prompt += "Addition: " + state["task"]
    logger.info(f"AnalyticREQUEST: {prompt}")
    response = llm.invoke([HumanMessage(content=prompt)]).content


    logger.info(f"AnalyticRESPONSE: {response}")
    return {
        "current_plan": response,
        "analytic_attempts": 1,
        "previous_results": [{"analytic_request": state["task"], "analytic_response": response}],
    } # in previous_results: analytic_request, analytic_response, worker_response

def analytic_next_step(state: DataWorkState) -> Dict[str, Any]:
    if state["analytic_attempts"] > state["analytic_max_attempts"]:
        return {"done": True}
    request_data = data_utils.get_initial_data_form_df(state["df"], state["path_to_comp_desc"], state["name_of_file"])
    history = ""
    for attempt in state["previous_results"]:
        history += "DataAnalytic Request: " + attempt["analytic_request"] + "\n"
        history += "DataWorker Response: " + attempt["worker_response"] + "\n"
    request_data["history"] = history

    prompt = (data_analytic_prompt.initial_prompt
              .format(**request_data))
    response = llm.invoke([HumanMessage(content=prompt)]).content
    satisfy_rate = json.loads(response)["satisfy_rate"]
    done = satisfy_rate > 0.9

    logger.info(f"AnalyticREQUEST: {prompt}")
    logger.info(f"AnalyticRESPONSE: {response}")
    return {
        "current_plan": response,
        "analytic_attempts": state["analytic_attempts"] + 1,
        "previous_results": state["previous_results"].extend({"analytic_response": response}),
        "done": done,
    }


def after_check(state: DataWorkState) -> str:
    if state["done"]:
        return "stop_analysis"
    return "continue"


def run_coder_wrapper(state: DataWorkState) -> Dict[str, Any]:
    task = state.get("current_plan")
    if task is None:
        logger.warning("[Wrapper] No current_plan found, cannot run coder.")
        return {"done": True}

    max_attempts = state.get("worker_max_attempts", 3)
    data_dir = state.get("data_dir", None)
    name_of_file = state.get("name_of_file", "train.csv")

    # Извлекаем чистую задачу из JSON
    try:
        if isinstance(task, str) and task.strip().startswith('{'):
            task_json = json.loads(task)
            task = task_json.get("data_planner_request", task)
    except json.JSONDecodeError:
        pass

    # Добавляем имя файла в задачу, если его нет
    if "train.csv" not in task and "train.csv" not in task.lower():
        task = f"Load and analyze file '{name_of_file}'. {task}"

    logger.info(f"[Wrapper] Running coder with task:\n{task}")

    result = run_coder(task, max_attempts, data_dir=data_dir)

    if result.get("execution_output"):
        worker_response = result["execution_output"]
    elif result.get("execution_error"):
        worker_response = f"Error: {result['execution_error']}"
    else:
        worker_response = "No output"

    logger.info(f"[Wrapper] Coder finished. Response:\n{worker_response}")

    # Исправлено: проверяем, что prev_results существует и это список
    prev_results = state.get("previous_results")
    if prev_results is None:
        prev_results = []

    if prev_results:
        # Обновляем последнюю запись
        updated_last = prev_results[-1].copy()
        updated_last["worker_response"] = worker_response
        prev_results = prev_results[:-1] + [updated_last]
    else:
        prev_results = [{"worker_response": worker_response}]

    logger.info(f"DataWorkerREQUEST: {task}")
    logger.info(f"DataWorkerRESPONSE: {worker_response}")

    return {
        "previous_results": prev_results,
        "worker_attempts": state.get("worker_attempts", 0) + 1,
    }


def run_data_worker(state: DataWorkState, max_attempts: int = 3) -> Dict[str, Any]:

    # Строим граф
    graph = StateGraph(DataWorkState)
    graph.add_node("start_analytic", analytic_initial_step)
    graph.add_node("run_coder", run_coder_wrapper)
    graph.add_node("continue_analytic", analytic_next_step)

    graph.set_entry_point("start_analytic")
    graph.add_edge("start_analytic", "run_coder")
    graph.add_conditional_edges("run_coder", after_check, {
        "stop_analysis": END,
        "continue": "continue_analytic"
    })
    graph.add_edge("continue_analytic", "run_coder")

    app = graph.compile()

    result = app.invoke(state)
    return result


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    initial_state = {
        "df": df,
        "data_dir": "data",
        "path_to_comp_desc": "data/KaggleDescription.txt",
        "name_of_file": "train.csv",

        "current_plan": None,
        "analytic_attempts": None,
        "analytic_max_attempts": 4,
        "satisfy_rate": None,

        "worker_attempts": None,
        "worker_max_attempts": 3,

        "previous_results": [],
        "done": False,
    }
    run_data_worker(initial_state)