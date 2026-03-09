import os
import json
from enum import Enum
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer, CrossEncoder
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
        local_model_folder = './models/saved/all-MiniLM-L6-v2'
        if os.path.exists(local_model_folder):
            self.embedding_model = SentenceTransformer(local_model_folder, cache_folder="./models")
        else:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="./models")
            self.embedding_model.save(local_model_folder)
        # Initialize reranker model
        local_model_folder = './models/saved/ms-marco-MiniLM-L-6-v2'
        if os.path.exists(local_model_folder):
            self.reranker = CrossEncoder(local_model_folder, cache_folder="./models")
        else:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', cache_folder="./models")
            self.reranker.save(local_model_folder)

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

        # Early exit if nothing returned
        if not results['ids'] or not results['ids'][0]:
            return []

        # Prepare candidates
        candidates = []
        for content, chunk_id, metadata, distance in zip(
            results['documents'][0],
            results['ids'][0],
            results['metadatas'][0],
            results['distances'][0]
        ):
            candidates.append({
                'content': content,
                'chunk_id': chunk_id,
                'metadata': metadata,
                'chunk_tags': json.loads(metadata.get('tags', '[]')),
                'vector_score': 1 - distance,
            })

        if not candidates:
            return []

        # Rerank with CrossEncoder using (query, content) pairs
        pairs = [(query, c['content']) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        # Attach rerank scores and sort by it (desc)
        for c, score in zip(candidates, rerank_scores):
            c['rerank_score'] = float(score)

        candidates.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Take top n_results and format
        top_candidates = candidates[:n_results]
        formatted_results = []
        for c in top_candidates:
            md = c['metadata']
            formatted_results.append({
                'content': c['content'],
                'chunk_id': c['chunk_id'],
                'source_id': md.get('source_id'),
                'source_title': md.get('source_title'),
                'source_url': md.get('source_url'),
                'chunk_type': md.get('chunk_type'),
                'content_type': md.get('content_type'),
                'tags': c['chunk_tags'],
                'similarity_score': c['vector_score'],
                'rerank_score': c['rerank_score'],
                'position': int(md.get('position', 0)),
                'metadata': {k: v for k, v in md.items()
                             if k not in ['chunk_id', 'source_id', 'source_title',
                                          'source_url', 'chunk_type', 'content_type',
                                          'tags', 'position']}
            })

        return formatted_results
