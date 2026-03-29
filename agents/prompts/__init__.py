# prompts/__init__.py
from .coder_prompts import (
    INITIAL_CODE_PROMPT,
    FIX_CODE_PROMPT,
    TRAINING_CODE_PROMPT
)
from .data_worker_prompts import DATA_ANALYSIS_PROMPT
from .supervisor_prompts import SUPERVISOR_PLAN_PROMPT, SUPERVISOR_ANALYSIS_PROMPT
from .validator_prompts import VALIDATION_PROMPT

__all__ = [
    "INITIAL_CODE_PROMPT",
    "FIX_CODE_PROMPT",
    "TRAINING_CODE_PROMPT",
    "DATA_ANALYSIS_PROMPT",
    "SUPERVISOR_PLAN_PROMPT",
    "SUPERVISOR_ANALYSIS_PROMPT",
    "VALIDATION_PROMPT"
]