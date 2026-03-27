import os
import subprocess
import ast
from typing import Dict, Any, TypedDict, Optional


def extract_code(text: str) -> str:
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    else:
        text = text.strip()
    return text


def clean_code(code: str) -> str:
    """Очищает и исправляет распространённые синтаксические ошибки в сгенерированном коде"""
    import re

    # 1. Удаляем все разрывы строк внутри слов (например, "as p\nd" -> "as pd")
    code = re.sub(r'(\w+)\n(\w+)', r'\1\2', code)
    code = re.sub(r'(\w+)\n\s*(\w+)', r'\1\2', code)

    # 2. Исправляем слитые импорты
    code = re.sub(r'import\s+(\w+)\s+as\s+(\w+)(?=[a-zA-Z])', r'import \1 as \2\n', code)
    code = re.sub(r'import(\w+)', r'import \1', code)
    code = re.sub(r'from\s+(\w+)\s+import\s+(\w+)', r'from \1 import \2', code)

    # 3. Исправляем слитые слова
    code = re.sub(r'numpyas\s+', 'numpy as ', code)
    code = re.sub(r'pandasaspd', 'pandas as pd', code)
    code = re.sub(r'pandasas\s+pd', 'pandas as pd', code)
    code = re.sub(r'pandas\s+aspd', 'pandas as pd', code)
    code = re.sub(r'seabornas\s+sns', 'seaborn as sns', code)
    code = re.sub(r'matplotlib\.pyplotas\s+plt', 'matplotlib.pyplot as plt', code)
    code = re.sub(r'impor\s+t', 'import', code)
    code = re.sub(r'as\s+p\s*d', 'as pd', code)
    code = re.sub(r'as\s+n\s*p', 'as np', code)
    code = re.sub(r'as\s+s\s*n\s*s', 'as sns', code)
    code = re.sub(r'as\s+p\s*l\s*t', 'as plt', code)

    # 4. Добавляем пробелы после запятых
    code = re.sub(r',(?=[^\s])', ', ', code)

    # 5. Исправляем комментарии, слитые с кодом
    code = re.sub(r'#([^\s])', r'# \1', code)

    # 6. Разделяем слитые строки
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Если строка не пустая и не комментарий
        if line.strip() and not line.strip().startswith('#'):
            # Проверяем на наличие нескольких операторов
            if '=' in line and any(op in line for op in ['for', 'if', 'while']):
                # Оставляем как есть
                cleaned_lines.append(line)
            elif line.count('=') > 1 and '=' not in line.split('#')[0].split(',')[0]:
                # Разделяем по =, но осторожно
                parts = re.split(r'(?<=[^=])\s+(?=\w+\s*=)', line)
                if len(parts) > 1:
                    cleaned_lines.extend(parts)
                else:
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    code = '\n'.join(cleaned_lines)

    # 7. Исправляем отступы после двоеточия
    lines = code.split('\n')
    cleaned_lines = []
    indent_level = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue

        # Уменьшаем отступ для except, finally, elif, else
        if stripped.startswith('except') or stripped.startswith('finally') or \
                stripped.startswith('elif') or stripped.startswith('else'):
            indent_level = max(0, indent_level - 1)

        # Добавляем строку с правильным отступом
        cleaned_lines.append('    ' * indent_level + stripped)

        # Увеличиваем отступ после блоков
        if stripped.endswith(':') and not stripped.startswith('#'):
            indent_level += 1

    code = '\n'.join(cleaned_lines)

    # 8. Удаляем лишние пустые строки
    code = re.sub(r'\n\s*\n\s*\n', '\n\n', code)

    # 9. Заменяем path_to_file.csv на train.csv
    code = re.sub(r'path_to_file\.csv', 'train.csv', code)
    code = re.sub(r"'path_to_file\.csv'", "'train.csv'", code)
    code = re.sub(r'"path_to_file\.csv"', '"train.csv"', code)

    return code


def execute_in_docker(code: str, timeout: int = 60, data_dir: Optional[str] = None) -> tuple[bool, str]:
    """Выполняет Python код в Docker-контейнере через -c с автоматической установкой библиотек"""
    if not code.strip():
        return False, "Code empty"

    # Проверяем, существует ли кастомный образ с предустановленными библиотеками
    try:
        subprocess.run(["docker", "image", "inspect", "python-ml:3.10-slim"],
                       capture_output=True, check=True)
        image = "python-ml:3.10-slim"
        full_code = code
    except subprocess.CalledProcessError:
        # Используем стандартный образ и устанавливаем библиотеки на лету
        image = "python:3.10-slim"
        setup_code = '''
import subprocess, sys, importlib

def install(pkg):
    try:
        importlib.import_module(pkg)
        return
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])

for pkg in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn']:
    install(pkg)
'''
        full_code = setup_code + code

    # Формируем команду Docker
    cmd = ["docker", "run", "--rm"]

    # Если указана директория с данными, монтируем её в /data
    if data_dir and os.path.exists(data_dir):
        abs_data_dir = os.path.abspath(data_dir)
        cmd.extend(["-v", f"{abs_data_dir}:/data", "-w", "/data"])

    cmd.extend([image, "python", "-c", full_code])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout or "No output"
        else:
            return False, result.stderr or result.stdout
    except subprocess.TimeoutExpired:
        return False, f"Execution timed out after {timeout} seconds"
    except FileNotFoundError:
        return False, "Docker not found. Please install Docker."
    except Exception as e:
        return False, f"Docker error: {str(e)}"


def check_syntax(state: TypedDict) -> Dict[str, Any]:
    code = state.get("current_code")
    if not code:
        return {"syntax_error": "Code is empty"}
    try:
        ast.parse(code)
        return {"syntax_error": None}
    except SyntaxError as e:
        return {"syntax_error": f"Syntax error: {e.msg} at line {e.lineno}"}

