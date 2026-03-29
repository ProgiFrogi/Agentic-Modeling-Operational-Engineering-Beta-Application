"""Trainer agent for model training and evaluation"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from utils import logger
from utils.session_manager import SessionManager
from agents.coder import run_coder
import pickle
from datetime import datetime
import joblib

class TrainerState(TypedDict):
    session: SessionManager
    model_type: str  # 'linear', 'tree', 'ensemble', 'xgboost', etc.
    target_column: str
    metric: str  # 'mse', 'mae', 'rmse'
    train_config: Dict[str, Any]
    model: Optional[Any]
    predictions: Optional[np.ndarray]
    scores: Dict[str, float]
    training_code: str
    execution_output: str
    errors: List[str]
    attempts: int
    max_attempts: int
    done: bool


llm = ChatOllama(
    model="qwen2.5-coder:14b-instruct-q4_K_M",
    temperature=0,
)


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from LLM response"""
    clean_response = response.strip()

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
        import re
        json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return {}


def generate_training_plan(state: TrainerState) -> Dict[str, Any]:
    """Generate training plan based on data"""

    # Get data info
    train_df = pd.read_csv(state['session'].train_path)
    test_df = pd.read_csv(state['session'].test_path)

    train_info = {
        "shape": train_df.shape,
        "columns": list(train_df.columns),
        "dtypes": {col: str(dtype) for col, dtype in train_df.dtypes.items()},
        "missing": train_df.isnull().sum().to_dict()
    }

    prompt = f"""
You are a machine learning engineer. Create a training plan for the given data.

Session directory: {state['session'].session_dir}
Target column: {state['target_column']}
Metric: {state['metric']}

Data info:
- Train shape: {train_info['shape']}
- Columns: {train_info['columns'][:10]}...
- Data types: {train_info['dtypes']}

Available models: {state['model_type']}


Rules:
- Use memory-efficient code
- Handle categorical features properly (frequency encoding for high cardinality)
- Scale numerical features if needed
- Use train/val split (80/20)
- Save model and predictions
- Print all metrics
- Handle missing values BEFORE training
- DO NOT use OneHotEncoder on columns with >50 unique values - use LabelEncoder or frequency encoding instead
- DO NOT use ColumnTransformer with complex pipelines - keep it simple
- Use pandas for missing value handling: fillna(median) for numerical, fillna('Unknown') for categorical
- Use LabelEncoder for categorical columns
- Use StandardScaler for numerical columns

Output ONLY the Python code, no explanations, no markdown, no example code.
"""

    response = llm.invoke([HumanMessage(content=prompt)]).content
    plan = extract_json_from_response(response)

    logger.info(f"Training plan generated: {plan}")

    return {
        "model_type": plan.get("model_type", state["model_type"]),
        "train_config": plan,
        "attempts": 1
    }


def generate_training_code(state: TrainerState) -> Dict[str, Any]:
    """Generate code for training model"""

    config = state.get("train_config", {})
    target_column = state.get("target_column", "target")
    metric = state.get("metric", "mse")

    code_prompt = f"""
Write Python code to train a model on the data.

IMPORTANT: You are ALREADY in the session directory. Use relative paths like 'train.csv', not absolute paths.

CRITICAL INFORMATION:
- Target column name is: 'target'
- Task type: regression
- Evaluation metric: MSE

Requirements:
1. Load data from: 'train.csv' and 'test.csv' (use relative paths)
2. Target column: 'target'
3. Task type: regression (metric: MSE)
4. Configuration: {json.dumps(config, indent=2)}
5. Save trained model to: 'model.pkl' (use joblib.dump)
6. Save predictions on test data to: 'predictions.csv'
7. Calculate and print model score on validation set
8. Print feature importance if applicable

Rules:
- Use memory-efficient code
- Use pandas fillna() for missing values (NOT sklearn imputers if possible)
- For categorical columns with >50 unique values, use frequency encoding
- For categorical columns with <=50 unique values, use LabelEncoder
- Scale numerical features with StandardScaler
- Use train/val split (80/20)
- Save model with joblib.dump
- Print all metrics
- DO NOT use .ravel() - y is already 1D
Save predictions on test data to: 'predictions.csv' with format: 'index,prediction'
   - The 'index' column should be the index of test data (starting from 0)
   - The 'prediction' column should be the predicted values
Output ONLY the Python code, no explanations.
"""

    result = run_coder(code_prompt, max_attempts=3, data_dir=str(state['session'].session_dir))

    return {
        "training_code": result.get("final_code") or result.get("current_code", ""),
        "execution_output": result.get("execution_output", ""),
        "errors": [result.get("execution_error")] if result.get("execution_error") else [],
        "done": result.get("execution_error") is None
    }


def load_model_results(state: TrainerState) -> Dict[str, Any]:
    """Load saved model and predictions"""

    session_dir = state['session'].session_dir

    # Load model - используем joblib для загрузки
    model_path = session_dir / "model.pkl"
    model = None
    if model_path.exists():
        try:
            # Сначала пробуем joblib
            import joblib
            model = joblib.load(model_path)
        except:
            try:
                # Если не получилось, пробуем pickle
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            except Exception as e:
                print(f"Error loading model: {e}")

    # Load predictions
    pred_path = session_dir / "predictions.csv"
    predictions = None
    if pred_path.exists():
        predictions = pd.read_csv(pred_path)

    # Load scores if saved
    scores_path = session_dir / "scores.json"
    scores = {}
    if scores_path.exists():
        with open(scores_path, 'r') as f:
            scores = json.load(f)

    # Если модель не загружена, но код выполнился успешно - считаем что все ок
    done = state.get("done", False) or (model is not None)

    return {
        "model": model,
        "predictions": predictions,
        "scores": scores,
        "done": done
    }


def after_training(state: TrainerState) -> str:
    """Determine next step after training"""
    if state["done"]:
        return "end"
    elif state["attempts"] < state["max_attempts"]:
        return "retry"
    else:
        return "fail"


def run_trainer(session: SessionManager,
                target_column: str = "target",
                metric: str = "mse",
                model_type: str = "auto",
                max_attempts: int = 3) -> Dict[str, Any]:
    """Run the trainer agent"""

    initial_state = {
        "session": session,
        "model_type": model_type,
        "target_column": target_column,
        "metric": metric,
        "train_config": {},
        "model": None,
        "predictions": None,
        "scores": {},
        "training_code": "",
        "execution_output": "",
        "errors": [],
        "attempts": 0,
        "max_attempts": max_attempts,
        "done": False
    }

    # Build graph
    graph = StateGraph(TrainerState)
    graph.add_node("plan", generate_training_plan)
    graph.add_node("train", generate_training_code)
    graph.add_node("load_results", load_model_results)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "train")
    graph.add_edge("train", "load_results")
    graph.add_conditional_edges("load_results", after_training, {
        "end": END,
        "retry": "train",
        "fail": END
    })

    app = graph.compile()

    result = app.invoke(initial_state)

    print("=" * 50)
    print("Training Results:")
    print(f"Model type: {result.get('model_type')}")
    print(f"Scores: {result.get('scores')}")
    print(f"Errors: {result.get('errors')}")

    return result