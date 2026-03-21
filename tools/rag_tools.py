"""RAG retrieval for agents (uses existing Chroma vector store)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def retrieve_code(
    query: str,
    n_results: int = 3,
    tags: Optional[List[str]] = None,
    vector_store_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Top code-oriented chunks from the Kaggle notebook index (embedding + rerank).
    Requires a built index (see rag/init.py or KaggleRAGPipeline.build_index_from_kaggle).
    """
    from config.settings import get_settings
    from rag.rag_types import ChunkType
    from rag.storage import VectorStore

    path = vector_store_path or get_settings().rag_vector_store_path
    store = VectorStore(persist_directory=path)
    raw = store.search_chunks(
        query=query,
        chunk_type=ChunkType.CODE_SNIPPET,
        tags=tags,
        n_results=n_results,
    )
    out: List[Dict[str, Any]] = []
    for r in raw:
        out.append(
            {
                "source_title": r.get("source_title"),
                "chunk_type": r.get("chunk_type"),
                "similarity_score": r.get("similarity_score"),
                "rerank_score": r.get("rerank_score"),
                "content": r.get("content"),
                "tags": r.get("tags"),
            }
        )
    return out


def retrieve_code_as_json(*args: Any, **kwargs: Any) -> str:
    return json.dumps(retrieve_code(*args, **kwargs), ensure_ascii=False, indent=2)
