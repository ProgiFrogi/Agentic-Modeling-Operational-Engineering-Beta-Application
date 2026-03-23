import subprocess
import ast
from typing import Dict, Any, TypedDict


def extract_code(text: str) -> str:
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    else:
        text = text.strip()
    return text

def execute_in_docker(code: str, timeout: int = 10) -> tuple[bool, str]:
    """Выполняет Python код в Docker-контейнере через -c"""
    if not code.strip():
        return False, "Code empty"
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "python:3.10-slim", "python", "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
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

