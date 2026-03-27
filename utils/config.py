import os
import yaml
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Конфигурация LLM моделей"""
    # OpenRouter (для сложных задач)
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Бесплатные модели OpenRouter (по приоритету)
    reasoning_model: str = "deepseek/deepseek-r1:free"  # Сложные рассуждения
    coding_model: str = "qwen/qwen-2.5-coder-32b-instruct:free"  # Генерация кода
    fast_model: str = "google/gemma-3-4b-it:free"  # Быстрые ответы

    # Локальные модели (для 3GB VRAM)
    use_local: bool = True
    local_model_path: str = "./models/phi-2.Q4_K_M.gguf"  # ~1.6GB
    local_context_length: int = 2048
    local_temperature: float = 0.7

    # Fallback стратегия
    fallback_to_openrouter: bool = True

@dataclass
class PipelineConfig:
    """Конфигурация пайплайна"""
    # Пути
    data_dir: str = "./data"
    output_dir: str = "./output"
    models_dir: str = "./models"

    # Ограничения
    max_features: int = 100
    max_models_to_try: int = 5
    cv_folds: int = 5
    early_stopping_rounds: int = 50

    # Качество
    min_cv_score: float = 0.0
    target_improvement: float = 0.01

    # Безопасность
    max_execution_time: int = 3600  # 1 час на итерацию
    safe_mode: bool = True  # Не выполнять произвольный код без проверки

class Config:
    """Главный класс конфигурации"""

    def __init__(self, config_path: str = "config.yaml"):
        self.model = ModelConfig()
        self.pipeline = PipelineConfig()
        self._load_from_yaml(config_path)

    def _load_from_yaml(self, path: str):
        """Загружает конфиг из YAML, перезаписывая дефолты"""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if 'model' in data:
                for key, value in data['model'].items():
                    if hasattr(self.model, key):
                        setattr(self.model, key, value)

            if 'pipeline' in data:
                for key, value in data['pipeline'].items():
                    if hasattr(self.pipeline, key):
                        setattr(self.pipeline, key, value)

    def get_openrouter_headers(self) -> Dict[str, str]:
        """Возвращает заголовки для OpenRouter API"""
        return {
            "Authorization": f"Bearer {self.model.openrouter_api_key}",
            "HTTP-Referer": "https://localhost",
            "X-Title": "Kaggle Multi-Agent System"
        }

# Глобальный инстанс
_config = None

def get_config(config_path: str = "config.yaml") -> Config:
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

# Конфигурация для Kaggle Multi-Agent System

