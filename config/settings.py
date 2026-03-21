from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_base_url: Optional[str] = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_planner_model: str = Field(default="gpt-4o", validation_alias="OPENAI_PLANNER_MODEL")

    competition_ref: str = Field(default="mws-ai-agents-2026", validation_alias="COMPETITION_REF")
    workspace_root: str = Field(default="./workspace", validation_alias="WORKSPACE_ROOT")
    # If set, used as the agent workspace instead of WORKSPACE_ROOT / <competition_slug>
    workspace_path: Optional[str] = Field(default=None, validation_alias="WORKSPACE_PATH")
    workflow_verbose: bool = Field(default=True, validation_alias="WORKFLOW_VERBOSE")
    rag_vector_store_path: str = Field(
        default="./rag/kaggle_vector_store",
        validation_alias="RAG_VECTOR_STORE_PATH",
    )

    code_execution_timeout: int = Field(default=120, validation_alias="CODE_EXECUTION_TIMEOUT")
    max_workflow_iterations: int = Field(default=50, validation_alias="MAX_WORKFLOW_ITERATIONS")
    # After this many planner steps, if submission.csv is still missing, skip more EDA/prep and run coder.
    force_coder_min_planner_iteration: int = Field(default=1, validation_alias="FORCE_CODER_MIN_PLANNER_ITERATION")


@lru_cache
def get_settings() -> Settings:
    return Settings()
