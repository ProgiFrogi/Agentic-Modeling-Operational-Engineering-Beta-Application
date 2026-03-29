import os
import subprocess
import ast
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, TypedDict, Optional, Tuple


def extract_code(text: str) -> str:
    if "```python" in text:
        text = text.split("```python")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    else:
        text = text.strip()
    return text

def check_syntax(state: TypedDict) -> Dict[str, Any]:
    code = state.get("current_code")
    if not code:
        return {"syntax_error": "Code is empty"}
    try:
        ast.parse(code)
        return {"syntax_error": None}
    except SyntaxError as e:
        return {"syntax_error": f"Syntax error: {e.msg} at line {e.lineno}"}


def execute_with_saving(code: str, data_dir: str | None = None,
                        output_dir: str | None = None) -> Tuple[bool, str, Optional[str]]:
    """
    Execute Python code and save execution results.

    Args:
        code: Python code to execute
        data_dir: Directory with data files
        output_dir: Directory to save execution results. If None, creates a timestamped folder

    Returns:
        Tuple[bool, str, Optional[str]]: (success, output/error, result_file_path)
    """
    from datetime import datetime
    import json

    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        # Create a timestamped directory in the current location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"execution_results_{timestamp}")
        output_path.mkdir(parents=True, exist_ok=True)

    # Determine working directory for subprocess
    working_dir = None
    if data_dir:
        data_dir_path = Path(data_dir)
        if data_dir_path.exists():
            working_dir = str(data_dir_path)
        else:
            return False, f"Data directory not found: {data_dir}", None

    # Prepare code without adding os.chdir
    # Instead, we'll rely on the subprocess cwd parameter
    code_to_execute = f"""
import os
import sys
import json
from pathlib import Path

# Add output directory to path for saving results
output_dir = r'{str(output_path)}'
os.makedirs(output_dir, exist_ok=True)

# Add data directory to sys.path if needed
data_dir = r'{data_dir}' if {data_dir is not None} else None
if data_dir and os.path.exists(data_dir):
    sys.path.insert(0, data_dir)
    print(f"Data directory available: {{data_dir}}")

# Print current working directory for debugging
print(f"Current working directory: {{os.getcwd()}}")
print(f"Files in current directory: {{os.listdir('.')[:10]}}")  # Show first 10 files

# Original code
{code}
"""

    # Create temporary file for execution
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_file_path = f.name
        f.write(code_to_execute)

    try:
        # Execute the code with working directory set
        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=working_dir  # This sets the working directory for the subprocess
        )

        # Save execution info
        execution_info = {
            "timestamp": datetime.now().isoformat(),
            "success": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "code": code,
            "data_dir": data_dir,
            "output_dir": str(output_path),
            "working_dir": working_dir
        }

        # Save execution info to JSON file
        info_file = output_path / "execution_info.json"
        with open(info_file, 'w') as f:
            json.dump(execution_info, f, indent=2)

        # Save code separately
        code_file = output_path / "executed_code.py"
        with open(code_file, 'w') as f:
            f.write(code)

        if result.returncode == 0:
            output = result.stdout.strip()
            if not output:
                output = f"Code executed successfully. Results saved to {output_path}"
            return True, output, str(output_path)
        else:
            error_msg = result.stderr.strip()
            if not error_msg:
                error_msg = f"Execution failed with return code {result.returncode}"
            return False, error_msg, str(output_path)

    except subprocess.TimeoutExpired:
        return False, "Execution timeout (30 seconds)", None
    except Exception as e:
        return False, f"Execution error: {str(e)}", None
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass