# agents/data_worker.py (улучшенный - реальная обработка данных)
"""Data Worker agent for data analysis and preprocessing"""

import json
import pandas as pd
from typing import Dict, Any, TypedDict, List
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

from utils import logger, SessionManager
from config import get_config
from agents.prompts import DATA_ANALYSIS_PROMPT
from agents.coder import run_coder


class DataWorkerState(TypedDict):
    session: SessionManager
    task: str
    satisfy_rate: float
    history: List[Dict[str, Any]]
    attempts: int
    max_attempts: int
    done: bool
    last_output: str


class DataWorkerAgent:
    """Агент для анализа и предобработки данных"""

    def __init__(self):
        self.config = get_config()
        self.llm = self.config.get_llm()

    def _analyze_data_quality(self, session_dir: Path) -> Dict[str, Any]:
        """Анализирует качество данных и возвращает метрики"""
        try:
            train_path = session_dir / "train.csv"
            if not train_path.exists():
                return {"quality_score": 0.0, "issues": ["No train data"]}

            df = pd.read_csv(train_path)

            # Рассчитываем метрики качества
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            duplicate_ratio = df.duplicated().sum() / df.shape[0] if df.shape[0] > 0 else 0

            # Проверяем наличие целевой переменной
            target_col = self.config.competition.target_column
            has_target = target_col in df.columns

            # Оценка качества (0-1)
            quality_score = 1.0
            issues = []

            if missing_ratio > 0.3:
                quality_score -= 0.3
                issues.append(f"High missing values ratio: {missing_ratio:.2%}")
            elif missing_ratio > 0.1:
                quality_score -= 0.1
                issues.append(f"Moderate missing values: {missing_ratio:.2%}")

            if duplicate_ratio > 0.1:
                quality_score -= 0.2
                issues.append(f"High duplicate ratio: {duplicate_ratio:.2%}")

            if not has_target:
                quality_score = 0
                issues.append(f"Target column '{target_col}' not found")

            return {
                "quality_score": max(0, quality_score),
                "issues": issues,
                "missing_ratio": missing_ratio,
                "duplicate_ratio": duplicate_ratio,
                "has_target": has_target,
                "shape": df.shape
            }

        except Exception as e:
            logger.error(f"Data quality analysis failed: {e}")
            return {"quality_score": 0.0, "issues": [str(e)]}

    def _generate_preprocessing_plan(self, state: DataWorkerState, quality_report: Dict[str, Any]) -> str:
        """Генерирует план предобработки данных"""

        session_dir = state["session"].session_dir
        train_path = session_dir / "train.csv"

        if not train_path.exists():
            return "Load the dataset and perform initial exploration"

        df = pd.read_csv(train_path)

        # Анализируем колонки
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = [c for c in df.columns if 'dt' in c.lower() or 'date' in c.lower()]

        # Формируем специфический план
        plan_parts = []

        if quality_report.get("missing_ratio", 0) > 0:
            plan_parts.append(f"Handle missing values: {quality_report['missing_ratio']:.2%} missing")

        if len(datetime_cols) > 0:
            plan_parts.append(f"Convert datetime columns: {datetime_cols}")

        if len(categorical_cols) > 0:
            high_cardinality = [c for c in categorical_cols if df[c].nunique() > 50]
            low_cardinality = [c for c in categorical_cols if df[c].nunique() <= 50]
            if high_cardinality:
                plan_parts.append(f"Use frequency encoding for high-cardinality columns: {high_cardinality}")
            if low_cardinality:
                plan_parts.append(f"Use label encoding for low-cardinality columns: {low_cardinality}")

        if len(numeric_cols) > 0:
            plan_parts.append(f"Scale numerical features: {numeric_cols[:5]}...")

        plan = "Perform the following data preprocessing:\n" + "\n".join(
            f"{i + 1}. {p}" for i, p in enumerate(plan_parts))
        plan += "\n\nAfter preprocessing, save the cleaned data back to train.csv and test.csv"

        return plan

    def run(self, session: SessionManager, max_attempts: int = 3) -> Dict[str, Any]:
        """Запускает агента обработки данных"""

        # Анализируем качество данных
        quality_report = self._analyze_data_quality(session.session_dir)

        # Если данные уже хорошего качества, пропускаем обработку
        if quality_report["quality_score"] > 0.8 and quality_report.get("missing_ratio", 1) < 0.05:
            logger.info(f"Data quality is good ({quality_report['quality_score']:.2f}), skipping preprocessing")
            return {
                "satisfy_rate": quality_report["quality_score"],
                "done": True,
                "quality_report": quality_report
            }

        # Генерируем план обработки
        preprocessing_plan = self._generate_preprocessing_plan(
            state={"session": session, "history": [], "attempts": 0},
            quality_report=quality_report
        )

        logger.info(f"Preprocessing plan: {preprocessing_plan[:200]}...")

        # Формируем задание для кодера
        coding_task = f"""
        Perform data preprocessing on the dataset.

        Session directory: {session.session_dir}

        Preprocessing requirements:
        {preprocessing_plan}

        Specific instructions:
        1. Load train.csv and test.csv
        2. Handle missing values appropriately:
           - Numerical: fill with median
           - Categorical: fill with mode or 'Unknown'
        3. Encode categorical variables:
           - Low cardinality (<=50): LabelEncoder
           - High cardinality (>50): frequency encoding
        4. Scale numerical features using StandardScaler
        5. Handle datetime columns if present (extract year, month, day, dayofweek)
        6. Save processed data back to train.csv and test.csv
        7. Print before/after shapes and memory usage

        The target column is '{self.config.competition.target_column}'.
        Make sure to preserve it in train data.
        """

        result = run_coder(
            task=coding_task,
            max_attempts=max_attempts,
            data_dir=str(session.session_dir),
            extra_rules="Focus on data preprocessing only, don't train models."
        )

        # Повторно анализируем качество после обработки
        new_quality_report = self._analyze_data_quality(session.session_dir)

        success = result.get("execution_error") is None
        satisfy_rate = new_quality_report["quality_score"] if success else 0.5

        logger.info(f"Data processing completed. New quality score: {satisfy_rate:.2f}")

        return {
            "satisfy_rate": satisfy_rate,
            "done": success and satisfy_rate > 0.7,
            "quality_report": new_quality_report,
            "preprocessing_plan": preprocessing_plan,
            "execution_output": result.get("execution_output", "")
        }


def run_data_worker(session: SessionManager, max_attempts: int = 3) -> Dict[str, Any]:
    agent = DataWorkerAgent()
    return agent.run(session, max_attempts)