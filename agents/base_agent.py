# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config import ConfigManager, get_config
from utils import logger
from utils.guardrails import InputValidator, CodeSafetyChecker


class BaseAgent(ABC):
    """Базовый класс для всех агентов с встроенными guardrails"""

    def __init__(self, config: Optional[ConfigManager] = None):
        self.config = config or get_config()
        self.llm = self.config.get_llm()
        self.input_validator = InputValidator(self.config)
        self.code_safety = CodeSafetyChecker(self.config)
        self.name = self.__class__.__name__

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Запуск агента"""
        pass

    def validate_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация входных данных"""
        if self.config.guardrails.enable_input_validation:
            return self.input_validator.validate(data)
        return {"valid": True, "errors": []}

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Проверка безопасности кода"""
        if self.config.guardrails.enable_code_safety:
            return self.code_safety.check(code)
        return True, ""

    def log(self, message: str, level: str = "info"):
        """Логирование с именем агента"""
        getattr(logger, level)(f"[{self.name}] {message}")