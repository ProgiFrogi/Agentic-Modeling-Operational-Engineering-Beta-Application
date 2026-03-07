from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


class ContentType(Enum):
    """Enum for content type classification"""
    NOTEBOOK = "notebook"
    DISCUSSION = "discussion"  # unsupported


class ChunkType(Enum):
    """Enum for chunk types within content"""
    MARKDOWN_CELL = "markdown_cell"
    # Code specific
    FUNCTION = "function"
    CLASS = "class"
    CODE_SNIPPET = "code_snippet"


class ChunkTags(Enum):
    """Predefined tags for Kaggle content"""
    # Competition types
    COMPUTER_VISION = "computer_vision"
    NLP = "nlp"
    TABULAR = "tabular"
    TIME_SERIES = "time_series"

    # Techniques
    DEEP_LEARNING = "deep_learning"
    MACHINE_LEARNING = "machine_learning"
    ENSEMBLE = "ensemble"
    FEATURE_ENGINEERING = "feature_engineering"
    EDA = "eda"
    CROSS_VALIDATION = "cross_validation"
    HYPERPARAMETERS = "hyperparameter_tuning"

    # Tools
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    KERAS = "keras"

    # Content type specific
    TUTORIAL = "tutorial"
    SOLUTION = "solution"
    DISCUSSION_TOPIC = "discussion_topic"
    QUESTION = "question"
    ANSWER = "answer"

    # Performance
    HIGH_SCORE = "high_score"
    ENSEMBLE_SOLUTION = "ensemble_solution"
    BENCHMARK = "benchmark"

    # Code specific
    INSTALLATION = "installation"
    IMPORTS = "imports"
    IMPLEMENTATION = "implementation"
    EVALUATION = "evaluation"
    VISUALIZATION = "visualization"
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    SPLIT = "train_test_split"


@dataclass
class ContentChunk:
    """Represents a single chunk of content from Kaggle"""
    id: str
    source_title: str
    chunk_type: ChunkType
    content_type: ContentType
    text: str
    code: Optional[str] = None
    code_description: Optional[str] = None
    tags: List[ChunkTags] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_size: int = 0


@dataclass
class KaggleSource:
    """Represents the original source (notebook or discussion)"""
    id: str
    title: str
    content_type: ContentType
    chunks: List[ContentChunk] = field(default_factory=list)
