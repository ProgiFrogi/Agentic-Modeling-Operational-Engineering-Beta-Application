import os
import json
from enum import Enum
from typing import List, Dict, Optional

# For embeddings and vector storage
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from rag.rag_types import ContentChunk, ContentType, ChunkType


class VectorStore:
    """Manage vector storage and retrieval of Kaggle content"""

    def __init__(self, persist_directory: str = "./kaggle_vector_store", encode_limit: int = 3000, overload_factor: int = 3):
        self.encode_limit = encode_limit
        self.overload_factor = overload_factor

        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="./models")

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collections
        self.chunks_collection = self._get_or_create_collection("kaggle_chunks")

    def _get_or_create_collection(self, name: str):
        """Get existing collection or create new one"""
        try:
            return self.chroma_client.get_collection(name)
        except NotFoundError:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

    def add_chunk(self, chunk: ContentChunk):
        """Add a single chunk to vector store"""
        if chunk.code_description:
            text_to_embed = chunk.code_description
        else:
            text_to_embed = chunk.text
        embedding = self.embedding_model.encode(text_to_embed[:self.encode_limit]).tolist()

        # Prepare metadata (ensure all values are strings or numbers)
        tag_values = [t.value if isinstance(t, Enum) else str(t) for t in (chunk.tags or [])]
        metadata = {
            "chunk_id": chunk.id,
            "source_title": str(chunk.source_title)[:100],
            "chunk_type": chunk.chunk_type.value,
            "content_type": chunk.content_type.value,
            "tags": json.dumps(tag_values)[:1000],
            "chunk_size": chunk.chunk_size,
        }

        # Add additional metadata with string conversion
        for k, v in chunk.metadata.items():
            if v is not None:
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
                else:
                    metadata[k] = str(v)[:100]

        # Add to ChromaDB
        self.chunks_collection.add(
            embeddings=[embedding],
            documents=[chunk.text],
            metadatas=[metadata],
            ids=[chunk.id]
        )

    def search_chunks(self, query: str, content_type: Optional[ContentType] = None,
                      chunk_type: Optional[ChunkType] = None,
                      tags: Optional[List[str]] = None,
                      tags_overlay_rule: str = '$and',
                      n_results: int = 10) -> List[Dict]:
        """
        Search for relevant chunks
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build filter
        where_filter = {}
        top_filter = []
        if content_type:
            top_filter.append({"content_type": {"$eq": content_type.value}})
        if chunk_type:
            top_filter.append({"chunk_type":  {"$eq": chunk_type.value}})
        if tags:
            tags_filter = []
            for tag in tags:
                tags_filter.append({"tags": {"$contains": tag}})
            top_filter.append({tags_overlay_rule: tags_filter})

        where_filter["$and"] = top_filter

        # Search
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * self.overload_factor,
            where=where_filter if top_filter else None
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for idx, (content, chunk_id, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['ids'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                # Parse tags
                chunk_tags = json.loads(metadata.get('tags', '[]'))

                # Filter by tags if specified
                if tags:
                    if not any(tag in chunk_tags for tag in tags):
                        continue

                formatted_results.append({
                    'content': content,
                    'chunk_id': chunk_id,
                    'source_id': metadata.get('source_id'),
                    'source_title': metadata.get('source_title'),
                    'source_url': metadata.get('source_url'),
                    'chunk_type': metadata.get('chunk_type'),
                    'content_type': metadata.get('content_type'),
                    'tags': chunk_tags,
                    'similarity_score': 1 - distance,
                    'position': int(metadata.get('position', 0)),
                    'metadata': {k: v for k, v in metadata.items()
                                 if k not in ['chunk_id', 'source_id', 'source_title',
                                              'source_url', 'chunk_type', 'content_type',
                                              'tags', 'position']}
                })

                if len(formatted_results) >= n_results:
                    break

        return formatted_results
