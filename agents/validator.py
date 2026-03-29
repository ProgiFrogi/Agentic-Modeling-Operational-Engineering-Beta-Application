"""Validator agent for evaluating model performance"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from utils import extract_json_from_response
from utils import logger
from utils.session_manager import SessionManager
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ValidatorState(TypedDict):
    session: SessionManager
    target_column: str
    metric: str
    threshold: float
    scores: Dict[str, float]
    validation_report: str
    passed: bool
    recommendations: List[str]
    done: bool

llm = ChatOllama(
    model="qwen2.5-coder:14b-instruct-q4_K_M",
    temperature=0,
)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Calculate specified metric"""
    if metric == "mse":
        return mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    else:
        return mean_squared_error(y_true, y_pred)

def validate_model(state: ValidatorState) -> Dict[str, Any]:
    """Validate trained model"""

    session_dir = state['session'].session_dir

    # Load predictions and actual values
    pred_path = session_dir / "predictions.csv"
    train_df = pd.read_csv(state['session'].train_path)

    if not pred_path.exists():
        return {
            "passed": False,
            "validation_report": "No predictions found",
            "scores": {},
            "recommendations": ["Train model first"],
            "done": True
        }

    predictions = pd.read_csv(pred_path)
    target = "target"

    if target not in train_df.columns:
        return {
            "passed": False,
            "validation_report": f"Target column '{target}' not found in train data",
            "scores": {},
            "recommendations": ["Check target column name"],
            "done": True
        }

    # Get actual values (last rows for validation)
    y_true = train_df[target].values

    # If predictions are for test set, we need validation split
    if len(predictions) != len(y_true):
        # Assume we have validation predictions
        y_pred = predictions.values.flatten() if len(predictions.shape) > 1 else predictions.values
    else:
        y_pred = predictions.values.flatten()

    # Calculate score
    score = calculate_metrics(y_true[:len(y_pred)], y_pred, state['metric'])

    scores = {
        state['metric']: score,
        "r2": r2_score(y_true[:len(y_pred)], y_pred) if len(y_pred) > 1 else 0
    }

    # Generate validation report via LLM
    prompt = f"""
You are a model validator. Evaluate the model performance:

Metric: MSE  
Score: {score}
Threshold: {state['threshold']}

Data shape: {len(y_pred)} validation samples

Generate a validation report with:
1. Is the model acceptable? (score <= threshold for MSE/RMSE, >= threshold for R2)
2. Recommendations for improvement if needed
3. Brief analysis of the result

Output in JSON format:
{{
    "passed": true/false,
    "analysis": "brief analysis",
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)]).content

    try:
        report_json = json.loads(response) if response.startswith('{') else extract_json_from_response(response)
        passed = report_json.get("passed", score <= state['threshold'] if state['metric'] in ['mse', 'rmse', 'mae'] else score >= state['threshold'])
        recommendations = report_json.get("recommendations", [])
        analysis = report_json.get("analysis", "")
    except:
        passed = score <= state['threshold'] if state['metric'] in ['mse', 'rmse', 'mae'] else score >= state['threshold']
        recommendations = []
        analysis = f"Score: {score}"

    validation_report = f"""
Validation Results:
- Metric: {state['metric']} = {score:.6f}
- R2 Score: {scores['r2']:.6f}
- Passed: {passed}

Analysis: {analysis}
"""

    return {
        "scores": scores,
        "validation_report": validation_report,
        "passed": passed,
        "recommendations": recommendations,
        "done": True
    }

def run_validator(session: SessionManager,
                  target_column: str = "target",
                  metric: str = "mse",
                  threshold: float = 0.1) -> Dict[str, Any]:
    """Run validator agent"""

    initial_state = {
        "session": session,
        "target_column": target_column,
        "metric": metric,
        "threshold": threshold,
        "scores": {},
        "validation_report": "",
        "passed": False,
        "recommendations": [],
        "done": False
    }

    graph = StateGraph(ValidatorState)
    graph.add_node("validate", validate_model)
    graph.set_entry_point("validate")
    graph.add_edge("validate", END)

    app = graph.compile()
    result = app.invoke(initial_state)

    print("="*50)
    print("Validation Results:")
    print(result.get("validation_report"))
    print(f"Recommendations: {result.get('recommendations')}")

    return result