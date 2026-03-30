import os
import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OllamaConfig:
    model: str = "qwen2.5-coder:14b-instruct-q4_K_M"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0


@dataclass
class OpenRouterConfig:
    model: str = "qwen/qwen-2.5-coder-32b-instruct:free"
    api_key: str = ""
    base_url: str = "https://openrouter.ai/api/v1"
    temperature: float = 0.0


@dataclass
class ModelConfig:
    provider: str = "ollama"  # ollama, openrouter
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)

    def get_llm(self):
        """Создает и возвращает LLM инстанс на основе конфигурации"""
        if self.provider == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=self.ollama.model,
                base_url=self.ollama.base_url,
                temperature=self.ollama.temperature,
            )
        elif self.provider == "openrouter":
            from langchain_openai import ChatOpenAI
            api_key = os.getenv("OPENROUTER_API_KEY")
            return ChatOpenAI(
                model=self.openrouter.model,
                api_key=api_key,
                base_url=self.openrouter.base_url,
                temperature=self.openrouter.temperature,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


@dataclass
class CompetitionFilesConfig:
    train: str = "train.csv"
    test: str = "test.csv"
    sample_submission: str = "sample_submission.csv"
    competition_info: str = "competition_info.txt"


@dataclass
class CompetitionConfig:
    name: str = "mws-ai-agents-2026"
    target_column: str = "target"
    metric: str = "mse"
    problem_type: str = "regression"
    files: CompetitionFilesConfig = field(default_factory=CompetitionFilesConfig)
    download_data: bool = False
    download_path: str = "./data"


@dataclass
class PipelineConfig:
    max_iterations: int = 3
    max_attempts_per_agent: int = 3
    execution_timeout: int = 60
    rag_retrievals: int = 3
    rag_char_limit: int = 3000
    safe_mode: bool = True
    data_dir: str = "./data"
    sessions_dir: str = "./sessions"
    logs_dir: str = "./logs"
    models_dir: str = "./models"
    benchmarks_dir: str = "./benchmarks"


@dataclass
class GuardrailsConfig:
    enable_input_validation: bool = True
    enable_code_safety: bool = True
    forbidden_imports: list = field(default_factory=lambda: [
        "os.system", "subprocess", "eval", "exec", "__import__", "open", "__builtins__"
    ])
    allowed_modules: list = field(default_factory=lambda: [
        "pandas", "numpy", "sklearn", "xgboost", "lightgbm", "catboost",
        "matplotlib", "seaborn", "scipy", "joblib", "pickle", "json", "csv"
    ])


class ConfigManager:
    """Главный менеджер конфигурации"""

    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.competition = CompetitionConfig()
        self.pipeline = PipelineConfig()
        self.guardrails = GuardrailsConfig()

        if config_path and Path(config_path).exists():
            self._load_from_yaml(config_path)

    def _load_from_yaml(self, path: str):
        """Загружает конфигурацию из YAML файла"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if 'model' in data:
            model_data = data['model']
            self.model.provider = model_data.get('provider', self.model.provider)

            if 'ollama' in model_data:
                self.model.ollama = OllamaConfig(**model_data['ollama'])
            if 'openrouter' in model_data:
                self.model.openrouter = OpenRouterConfig(**model_data['openrouter'])

        if 'competition' in data:
            comp_data = data['competition']
            self.competition.name = comp_data.get('name', self.competition.name)
            self.competition.target_column = comp_data.get('target_column', self.competition.target_column)
            self.competition.metric = comp_data.get('metric', self.competition.metric)
            self.competition.problem_type = comp_data.get('problem_type', self.competition.problem_type)
            self.competition.download_data = comp_data.get('download_data', self.competition.download_data)
            self.competition.download_path = comp_data.get('download_path', self.competition.download_path)

            if 'files' in comp_data:
                self.competition.files = CompetitionFilesConfig(**comp_data['files'])

        if 'pipeline' in data:
            for key, value in data['pipeline'].items():
                if hasattr(self.pipeline, key):
                    setattr(self.pipeline, key, value)

        if 'guardrails' in data:
            for key, value in data['guardrails'].items():
                if hasattr(self.guardrails, key):
                    setattr(self.guardrails, key, value)

    def get_llm(self):
        """Возвращает настроенную LLM"""
        return self.model.get_llm()


# Глобальный экземпляр
_config: Optional[ConfigManager] = None


def get_config(config_path: str = "config/config.yaml") -> ConfigManager:
    """Возвращает глобальный экземпляр конфигурации"""
    global _config
    if _config is None:
        _config = ConfigManager(config_path)
    return _config
