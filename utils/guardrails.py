import ast
import re
from typing import Dict, Any, Tuple
from config import ConfigManager


class InputValidator:
    """Валидатор входных данных для агентов"""

    def __init__(self, config: ConfigManager):
        self.config = config

    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует входные данные"""
        errors = []

        # Проверка на пустые значения
        for key, value in data.items():
            if value is None:
                errors.append(f"Field '{key}' is None")
            elif isinstance(value, str) and not value.strip():
                errors.append(f"Field '{key}' is empty string")

        # Проверка на опасные паттерны в строках
        dangerous_patterns = [
            r"__import__\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
            r"os\.system",
            r"subprocess\.",
        ]

        for key, value in data.items():
            if isinstance(value, str):
                for pattern in dangerous_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(f"Field '{key}' contains dangerous pattern: {pattern}")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


class CodeSafetyChecker:
    """Проверка безопасности кода перед выполнением"""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.forbidden_imports = config.guardrails.forbidden_imports
        self.allowed_modules = config.guardrails.allowed_modules + [
            "sklearn.preprocessing", "sklearn.impute", "sklearn.compose",
            "sklearn.pipeline", "sklearn.model_selection", "sklearn.ensemble",
            "sklearn.tree", "sklearn.linear_model", "sklearn.metrics", "joblib", "json"
        ]

    def check(self, code: str) -> Tuple[bool, str]:
        """Проверяет код на безопасность"""
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Проверка опасных вызовов
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        func_name = f"{node.func.value.id}.{node.func.attr}" if hasattr(node.func.value, 'id') else str(
                            node.func)
                        if func_name in self.forbidden_imports:
                            return False, f"Forbidden function call: {func_name}"

                # Проверка импортов
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name
                        if not any(module_name.startswith(allowed) for allowed in self.allowed_modules):
                            return False, f"Forbidden import: {module_name}"

                elif isinstance(node, ast.ImportFrom):
                    # Разрешаем импорты из разрешённых модулей/пакетов
                    module = node.module  # например, "sklearn.impute"

                    # Проверяем, что модуль начинается с разрешённого
                    is_allowed = False
                    if module:
                        for allowed in self.allowed_modules:
                            if module.startswith(allowed):
                                is_allowed = True
                                break

                    if not is_allowed:
                        return False, f"Forbidden import from: {module}"

            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error in code: {e}"
        except Exception as e:
            return False, f"Code safety check failed: {e}"
