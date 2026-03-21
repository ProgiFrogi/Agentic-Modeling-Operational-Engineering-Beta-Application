"""Competition-specific defaults for mws-ai-agents-2026 (tabular regression, MSE)."""

from pathlib import Path
from typing import List

DEFAULT_COMPETITION_REF = "mws-ai-agents-2026"

# Expected data files after setup_competition
DATA_FILES: List[str] = ["train.csv", "test.csv", "sample_submission.csv"]


def competition_workspace(workspace_root: str, competition_ref: str) -> Path:
    slug = competition_ref.split("/")[-1]
    return Path(workspace_root) / slug


def data_paths(workspace: Path) -> dict:
    return {
        "train": str(workspace / "train.csv"),
        "test": str(workspace / "test.csv"),
        "sample_submission": str(workspace / "sample_submission.csv"),
    }
