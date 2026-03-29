""" Analyze data and give instructions to data_worker """


import os
from typing import Dict, Any, TypedDict, List, Optional

import json
import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from utils import data_utils, logger
from dotenv import load_dotenv
from agents.prompts import data_analytic_prompt
from utils.session_manager import  SessionManager
from agents.coder import run_coder

load_dotenv()

class DataWorkState(TypedDict):
    task: str
    session: SessionManager
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


llm = ChatOllama(
    model="qwen2.5-coder:14b-instruct-q4_K_M",
    temperature=0,
)

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response, removing markdown formatting"""
    clean_response = response.strip()

    # Remove markdown code blocks if present
    if clean_response.startswith("```json"):
        clean_response = clean_response.split("```json")[1]
    elif clean_response.startswith("```"):
        clean_response = clean_response.split("```")[1]

    if clean_response.endswith("```"):
        clean_response = clean_response.rsplit("```", 1)[0]

    clean_response = clean_response.strip()

    try:
        return json.loads(clean_response)
    except json.JSONDecodeError:
        # Try to find JSON object using regex
        import re
        json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}


def get_dataset_info(session: SessionManager, name: str = "train") -> Dict:
    """Получает информацию о датасете из сессионной папки"""
    file_path = session.session_dir / f"{name}.csv"

    if not file_path.exists():
        return {"exists": False}

    df = pd.read_csv(file_path)
    return {
        "exists": True,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "head": df.head(5).to_dict()
    }


def analytic_initial_step(state: DataWorkState) -> Dict[str, Any]:
    # Получаем информацию о текущих файлах
    train_info = get_dataset_info(state["session"], "train")
    test_info = get_dataset_info(state["session"], "test")

    # Читаем описание
    with open(state["path_to_comp_desc"], 'r', encoding='utf-8') as f:
        description = f.read()

    request_data = {
        "description": description,
        "train_info": train_info,
        "test_info": test_info,
        "history": "It's your first try"
    }

    prompt = f"""
    You are a professional data analyst working with files in the session directory.

    Current session directory: {state['session'].session_dir}
    Files available: {state['session'].list_files()}

    Dataset info:
    Train data: {train_info['shape'][0]} rows, {train_info['shape'][1]} columns
    Test data: {test_info['shape'][0]} rows, {test_info['shape'][1]} columns

    Dataset description from author: 
    {description}

    History:
    {request_data['history']}

    IMPORTANT:
    - All your operations should work with files in the session directory
    - You can modify train.csv and test.csv directly
    - The files will be automatically saved after each operation
    - Use the session directory for all file operations
    - The data worker will execute code in the session directory

    Rate your confidence (0-1) that the data is ready for model training.
    If rate > 0.9, we proceed to model training.

    Give response in json with fields:
    - data_planner_request: str with commands for data_worker
    - satisfy_rate: float
    """

    logger.info(f"AnalyticREQUEST: {prompt}")
    response = llm.invoke([HumanMessage(content=prompt)]).content

    response_json = extract_json_from_response(response)
    data_planner_request = response_json.get("data_planner_request", "Process the data")
    satisfy_rate = response_json.get("satisfy_rate", 0.8)

    logger.info(f"AnalyticRESPONSE: {response}")

    return {
        "current_plan": data_planner_request,
        "satisfy_rate": satisfy_rate,
        "analytic_attempts": 1,
        "previous_results": [{
            "analytic_request": state["task"],
            "analytic_response": response,
            "worker_response": ""
        }],
    }


def analytic_next_step(state: DataWorkState) -> Dict[str, Any]:
    if state["analytic_attempts"] > state["analytic_max_attempts"]:
        return {"done": True}

    # Получаем актуальную информацию о файлах
    train_info = get_dataset_info(state["session"], "train")
    test_info = get_dataset_info(state["session"], "test")

    # Читаем описание
    with open(state["path_to_comp_desc"], 'r', encoding='utf-8') as f:
        description = f.read()

    # Формируем историю с ошибками и успехами
    history = ""
    errors = []
    successes = []

    for i, attempt in enumerate(state["previous_results"]):
        history += f"Step {i + 1}:\n"
        history += "  Request: " + attempt.get("analytic_request", "")[:200] + "\n"

        worker_response = attempt.get("worker_response", "")
        if "Error:" in worker_response:
            history += "  Status: FAILED\n"
            history += "  Error: " + worker_response[:200] + "\n"
            errors.append(worker_response[:200])
        else:
            history += "  Status: SUCCESS\n"
            history += "  Output: " + worker_response[:200] + "\n"
            successes.append(worker_response[:200])

    # Проверяем, есть ли успешные выполнения
    has_successful_run = len(successes) > 0

    # Формируем секцию с рекомендациями
    recommendations = ""
    if has_successful_run:
        recommendations = """
    DATA HAS BEEN SUCCESSFULLY PROCESSED!

    Based on the successful execution, the data now has:
    - Consistent data types between train and test
    - Missing values handled
    - Categorical variables encoded
    - Numerical features scaled

    If you are confident the data is ready for model training, set satisfy_rate > 0.9.
    """
    elif errors:
        recommendations = f"""
    PREVIOUS ERRORS (NEED TO FIX):
    {chr(10).join([f"  - {e}" for e in errors[-2:]])}

    The code failed. Your next instructions should specifically address these errors.
    """

    prompt = f"""
You are a professional data analyst working with files in the session directory.

Current session directory: {state['session'].session_dir}
Files available: {state['session'].list_files()}

Current dataset state:
Train data: {train_info['shape'][0]} rows, {train_info['shape'][1]} columns
Test data: {test_info['shape'][0]} rows, {test_info['shape'][1]} columns

Dataset description from author: 
{description}

HISTORY OF PREVIOUS ATTEMPTS:
{history}

{recommendations}

TASK: {state['task']}

CRITICAL INSTRUCTIONS:
1. If data has been successfully processed (no errors in last attempt), you should set satisfy_rate > 0.9 to proceed to model training
2. If there were errors, provide SPECIFIC instructions to fix them
3. DO NOT repeat the same instructions if they already succeeded

Rate your confidence (0-1) that the data is ready for model training.
If rate > 0.9, we proceed to model training.

Give response in json with fields:
- data_planner_request: str with commands for data_worker (empty if rate > 0.9)
- satisfy_rate: float
"""

    response = llm.invoke([HumanMessage(content=prompt)]).content

    response_json = extract_json_from_response(response)
    data_planner_request = response_json.get("data_planner_request", "")
    satisfy_rate = response_json.get("satisfy_rate", 0.0)

    done = satisfy_rate >= 0.9
    print(f"Done: {done}")
    logger.info(f"AnalyticREQUEST: {prompt[:500]}...")
    logger.info(f"AnalyticRESPONSE: {response}")
    logger.info(f"Parsed satisfy_rate: {satisfy_rate}, done: {done}")

    # Обновляем историю
    prev_results = state.get("previous_results", [])
    current_plan = state.get("current_plan", "")

    prev_results.append({
        "analytic_request": current_plan,
        "analytic_response": response,
        "worker_response": "",
        "analytic_attempt": state["analytic_attempts"]
    })

    return {
        "current_plan": data_planner_request if not done else "",
        "analytic_attempts": state["analytic_attempts"] + 1,
        "previous_results": prev_results,
        "done": done,
        "satisfy_rate": satisfy_rate,
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

    # Задача для кодера - работа с файлами в сессионной папке
    enhanced_task = f"""
    Work with files in the session directory: {state['session'].session_dir}

    Available files: {state['session'].list_files()}

    Task: {task}

    Instructions:
    1. Load data from the session directory files (train.csv, test.csv)
    2. Perform required transformations
    3. Save the processed data back to the same files (overwrite)
    4. You can create additional files if needed (they will be saved in the session directory)

    Important:
    - All file operations should use paths relative to the session directory
    - The session directory is: {state['session'].session_dir}
    - You can use: pd.read_csv('train.csv') - it will work from the session directory
    - After processing, save back using: df.to_csv('train.csv', index=False)

    The system will automatically use the updated files for the next steps.
    """

    logger.info(f"[Wrapper] Running coder with task:\n{enhanced_task}")

    # Запускаем кодера с указанием сессионной папки как data_dir
    result = run_coder(
        enhanced_task,
        max_attempts,
        data_dir=str(state['session'].session_dir)
    )

    if result.get("execution_output"):
        worker_response = result["execution_output"]
    elif result.get("execution_error"):
        worker_response = f"Error: {result['execution_error']}"
    else:
        worker_response = "No output"

    logger.info(f"[Wrapper] Coder finished.")
    logger.info(f"[Wrapper] Response: {worker_response[:500]}...")  # Первые 500 символов

    # Обновляем историю
    prev_results = state.get("previous_results", [])
    if prev_results:
        prev_results[-1]["worker_response"] = worker_response

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
    graph.add_edge("run_coder", "continue_analytic")
    graph.add_conditional_edges("continue_analytic", after_check, {
        "stop_analysis": END,
        "continue": "continue_analytic"
    })


    app = graph.compile()

    result = app.invoke(state)
    return result


if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    initial_state = {
        "df": df,
        "data_dir": "data",
        "path_to_comp_desc": "data/competition_info.txt",
        "name_of_file": "train.csv",

        "current_plan": None,
        "analytic_attempts": 0,
        "analytic_max_attempts": 4,
        "satisfy_rate": 0.0,

        "worker_attempts": 0,
        "worker_max_attempts": 3,

        "previous_results": [],
        "done": False,
    }
    run_data_worker(initial_state)