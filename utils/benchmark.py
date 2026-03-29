# utils/benchmark.py
"""Benchmarking utilities for model comparison"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score


class Benchmark:
    """Хранит и сравнивает метрики моделей"""

    def __init__(self, benchmark_dir: str = "./benchmarks"):
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def add_result(self, model_name: str, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        """Добавляет результат модели в бенчмарк"""
        result = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "metadata": metadata or {}
        }
        self.results.append(result)
        self._save()

    def _save(self):
        """Сохраняет результаты в JSON"""
        save_path = self.benchmark_dir / "benchmark_results.json"
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)

    def load(self):
        """Загружает результаты из JSON"""
        load_path = self.benchmark_dir / "benchmark_results.json"
        if load_path.exists():
            with open(load_path, 'r') as f:
                self.results = json.load(f)

    def get_best_model(self, metric: str, lower_is_better: bool = True) -> Optional[Dict[str, Any]]:
        """Возвращает лучшую модель по заданной метрике"""
        if not self.results:
            return None

        valid_results = [r for r in self.results if metric in r["metrics"]]
        if not valid_results:
            return None

        return min(valid_results, key=lambda x: x["metrics"][metric]) if lower_is_better \
            else max(valid_results, key=lambda x: x["metrics"][metric])

    def compare_models(self) -> pd.DataFrame:
        """Создаёт DataFrame для сравнения моделей"""
        data = []
        for result in self.results:
            row = {
                "model_name": result["model_name"],
                "timestamp": result["timestamp"]
            }
            row.update(result["metrics"])
            data.append(row)

        return pd.DataFrame(data)


class MetricsCalculator:
    """Калькулятор метрик для разных типов задач"""

    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Рассчитывает метрики для регрессии"""
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Рассчитывает метрики для классификации"""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "weighted")
        }

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, problem_type: str) -> Dict[str, float]:
        """Рассчитывает метрики в зависимости от типа задачи"""
        if problem_type == "regression":
            return MetricsCalculator.calculate_regression_metrics(y_true, y_pred)
        else:
            return MetricsCalculator.calculate_classification_metrics(y_true, y_pred)