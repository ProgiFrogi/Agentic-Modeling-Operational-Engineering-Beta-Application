from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

class KaggleState(TypedDict):
    """
    Глобальное состояние системы. Передается между всеми узлами графа.
    """
    # Метаданные соревнования
    competition_name: str
    problem_type: str  # 'classification', 'regression', 'multiclass'
    evaluation_metric: str
    target_column: str

    # Данные
    train_path: Optional[str]
    test_path: Optional[str]
    sample_submission_path: Optional[str]

    # DataFrames (в реальности будем хранить пути или hash)
    train_data: Optional[pd.DataFrame]
    test_data: Optional[pd.DataFrame]

    # Результаты EDA
    eda_report: Dict[str, Any]
    data_summary: Dict[str, Any]
    column_types: Dict[str, str]
    missing_values: Dict[str, int]

    # Feature Engineering
    feature_plan: List[str]
    engineered_features: List[str]
    feature_importance: Dict[str, float]

    # Моделирование
    model_configs: List[Dict[str, Any]]
    trained_models: Dict[str, Any]
    cross_val_scores: Dict[str, float]
    best_model_name: Optional[str]

    # Валидация
    validation_reports: List[Dict[str, Any]]
    critic_feedback: List[str]

    # Сабмишн
    submission_df: Optional[pd.DataFrame]
    submission_path: Optional[str]

    # Логи и ошибки
    logs: List[str]
    errors: List[str]
    current_step: str

    # Для итеративного улучшения
    iteration_count: int
    max_iterations: int
    improvement_history: List[float]


def create_initial_state(competition_name: str = "") -> KaggleState:
    """Создает начальное состояние"""
    return {
        "competition_name": competition_name,
        "problem_type": "",
        "evaluation_metric": "",
        "target_column": "",
        "train_path": None,
        "test_path": None,
        "sample_submission_path": None,
        "train_data": None,
        "test_data": None,
        "eda_report": {},
        "data_summary": {},
        "column_types": {},
        "missing_values": {},
        "feature_plan": [],
        "engineered_features": [],
        "feature_importance": {},
        "model_configs": [],
        "trained_models": {},
        "cross_val_scores": {},
        "best_model_name": None,
        "validation_reports": [],
        "critic_feedback": [],
        "submission_df": None,
        "submission_path": None,
        "logs": [],
        "errors": [],
        "current_step": "initialization",
        "iteration_count": 0,
        "max_iterations": 3,
        "improvement_history": []
    }


def log_state(state: KaggleState, message: str, level: str = "info"):
    """Добавляет лог в состояние"""
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level.upper()}] {message}"
    state["logs"].append(log_entry)

    if level == "error":
        state["errors"].append(log_entry)

    # Ограничиваем размер логов
    if len(state["logs"]) > 1000:
        state["logs"] = state["logs"][-1000:]

    return state

