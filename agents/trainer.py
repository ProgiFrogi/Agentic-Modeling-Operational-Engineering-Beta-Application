"""Trainer agent for model training and evaluation"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, TypedDict, List, Optional
from pathlib import Path

from utils import logger
from utils.session_manager import SessionManager
from agents.coder import run_coder
from config import get_config
from agents.prompts import TRAINING_CODE_PROMPT


class TrainerState(TypedDict):
    session: SessionManager
    model_type: str
    target_column: str
    metric: str
    problem_type: str
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


class TrainerAgent:
    """Агент для обучения моделей"""

    def __init__(self):
        self.config = get_config()
        self.llm = self.config.get_llm()

    def _extract_score_from_output(self, output: Optional[str], metric: str) -> Optional[float]:
        """Извлекает значение метрики из вывода кода"""
        import re

        if not output:
            return None

        # Ищем паттерны типа "MSE: 0.123" или "Validation MSE: 0.123"
        patterns = [
            rf"{metric}:\s*([0-9.]+)",
            rf"Validation\s+{metric}:\s*([0-9.]+)",
            rf"{metric.upper()}:\s*([0-9.]+)",
            rf"Score:\s*([0-9.]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass

        # Если не нашли, пытаемся найти любое число после ключевых слов
        keywords = ["mse", "rmse", "mae", "r2", "score"]
        for kw in keywords:
            pattern = rf"{kw}.*?([0-9.]+)"
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except:
                    pass

        return None

    def _calculate_scores_from_predictions(self, session_dir: Path, target_column: str, metric: str) -> Dict[
        str, float]:
        """Рассчитывает метрики на основе предсказаний и реальных значений"""
        try:
            # Загружаем train данные
            train_path = session_dir / "train.csv"
            predictions_path = session_dir / "predictions.csv"

            if not train_path.exists() or not predictions_path.exists():
                return {}

            train_df = pd.read_csv(train_path)
            predictions_df = pd.read_csv(predictions_path)

            if target_column not in train_df.columns:
                return {}

            # Берем последние N строк для валидации (20%)
            y_true = train_df[target_column].values
            val_size = int(len(y_true) * 0.2)
            y_true_val = y_true[-val_size:] if val_size > 0 else y_true

            # Получаем предсказания
            if 'prediction' in predictions_df.columns:
                y_pred = predictions_df['prediction'].values[:len(y_true_val)]
            else:
                # Если колонка называется по-другому
                pred_col = [c for c in predictions_df.columns if 'pred' in c.lower() or 'target' in c.lower()]
                if pred_col:
                    y_pred = predictions_df[pred_col[0]].values[:len(y_true_val)]
                else:
                    y_pred = predictions_df.iloc[:, -1].values[:len(y_true_val)]

            # Рассчитываем метрики
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

            # Конвертируем numpy типы в Python типы для JSON сериализации
            mse = float(mean_squared_error(y_true_val, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_true_val, y_pred))
            r2 = float(r2_score(y_true_val, y_pred))

            scores = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }

            logger.info(f"[Trainer] Calculated scores: {scores}")

            # Сохраняем scores в файл
            scores_path = session_dir / "scores.json"
            with open(scores_path, 'w') as f:
                json.dump(scores, f, indent=2)

            return scores

        except Exception as e:
            logger.error(f"[Trainer] Failed to calculate scores: {e}")
            return {}

    def run(
        self,
        session: SessionManager,
        max_attempts: int = 3,
        training_iteration: int = 0,
        improvement_context: str = "",
        previous_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Запускает агента-тренера"""

        target_column = self.config.competition.target_column
        metric = self.config.competition.metric
        problem_type = self.config.competition.problem_type

        code_prompt = TRAINING_CODE_PROMPT.format(
            target_column=target_column,
            problem_type=problem_type,
            metric=metric,
            config=json.dumps({}, indent=2)
        )

        extra_rules = f"""
        Remember:
        - Target column is '{target_column}'
        - Task type: {problem_type}
        - Evaluation metric: {metric}
        - Print the {metric} score clearly in the output
        - After training, save scores to 'scores.json' file
        """

        if training_iteration > 0:
            prev = json.dumps(previous_scores or {}, indent=2)
            plan = improvement_context.strip() or "(no structured plan — still change model/features vs previous iteration)"
            extra_rules += f"""

        REFINEMENT — training iteration {training_iteration} (not the first run):
        - Previous scores on the last attempt: {prev}
        - Supervisor improvement plan (follow concretely; do not repeat the same pipeline as iteration 0):
        {plan}
        - You must materially change the approach: different model family and/or features and/or preprocessing
          if the plan suggests it; do not output near-duplicate code to a naive baseline.
        """

        result = run_coder(
            code_prompt,
            max_attempts=max_attempts,
            data_dir=str(session.session_dir),
            extra_rules=extra_rules
        )

        # Извлекаем score из вывода (при падении кодера может быть None)
        execution_output = result.get("execution_output") or ""
        extracted_score = self._extract_score_from_output(execution_output, metric)

        # Рассчитываем scores из предсказаний
        scores = self._calculate_scores_from_predictions(session.session_dir, target_column, metric)

        # Если не удалось рассчитать, используем извлеченный
        if not scores and extracted_score is not None:
            scores = {metric: extracted_score}

        # Сохраняем результаты в сессию
        if scores:
            session.save_metadata("training_results", {
                "scores": scores,
                "success": result.get("execution_error") is None,
                "attempts": result.get("attempts", 0)
            })

        final_code = result.get("final_code") or result.get("current_code", "")
        if final_code and result.get("execution_error") is None:
            code_path = session.session_dir / "training_code.py"
            code_path.write_text(final_code, encoding="utf-8")
            logger.info(f"[Trainer] Saved training script to {code_path}")

        print("=" * 50)
        print("Training Results:")
        print(f"Scores: {scores}")
        print(f"Success: {result.get('execution_error') is None}")

        return {
            "scores": scores,
            "training_code": result.get("final_code") or result.get("current_code", ""),
            "execution_output": execution_output,
            "errors": [result.get("execution_error")] if result.get("execution_error") else [],
            "done": result.get("execution_error") is None,
            "attempts": result.get("attempts", 0)
        }


def run_trainer(
    session: SessionManager,
    max_attempts: int = 3,
    training_iteration: int = 0,
    improvement_context: str = "",
    previous_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    agent = TrainerAgent()
    return agent.run(
        session,
        max_attempts,
        training_iteration=training_iteration,
        improvement_context=improvement_context,
        previous_scores=previous_scores,
    )