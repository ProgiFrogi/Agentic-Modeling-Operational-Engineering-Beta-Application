"""
Kaggle RAG System for Notebooks and Discussions
This system extracts, processes, and indexes Kaggle content for semantic search.
"""

import os
import json
import hashlib
import uuid

import requests
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

# For code understanding
import ast
import inspect

# For embeddings and vector storage
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# For text processing
import re
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)
from langchain_text_splitters.base import Document


class ContentType(Enum):
    """Enum for content type classification"""
    CODE = "code"
    DISCUSSION = "discussion"

class ChunkType(Enum):
    """Enum for chunk types within content"""
    CODE_CELL = "code_cell"
    MARKDOWN_CELL = "markdown_cell"
    DISCUSSION_POST = "discussion_post"
    COMMENT = "comment"
    FUNCTION = "function"
    CLASS = "class"
    CODE_SNIPPET = "code_snippet"
    EXPLANATION = "explanation"
    QUESTION = "question"
    ANSWER = "answer"


class KaggleTag(Enum):
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

    # Tools
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

    # Content type specific
    TUTORIAL = "tutorial"
    SOLUTION = "solution"
    DISCUSSION_TOPIC = "discussion_topic"
    QUESTION = "question"
    ANSWER = "answer"

    # Data types
    TEXT = "text"
    NUMERIC = "numeric"

    # Performance
    HIGH_SCORE = "high_score"
    ENSEMBLE_SOLUTION = "ensemble_solution"
    BENCHMARK = "benchmark"

tags_all = [tag for tag in KaggleTag]
tags_all_str = [tag.value for tag in KaggleTag]

@dataclass
class ContentChunk:
    """Represents a single chunk of content from Kaggle"""
    id: str
    source_id: str
    source_title: str
    source_url: str
    chunk_type: ChunkType
    content_type: ContentType
    text: str
    code: Optional[str] = None
    code_description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: int = 0
    chunk_size: int = 0


@dataclass
class KaggleContent:
    """Data class for storing processed Kaggle content"""
    id: str
    title: str
    url: str
    content_type: ContentType
    raw_content: str
    processed_text: str
    code_description: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class KaggleSource:
    """Represents the original source (notebook or discussion)"""
    id: str
    title: str
    url: str
    content_type: ContentType
    author: str
    date: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunks: List[str] = field(default_factory=list)

class KaggleExtractor:
    """Extract content from Kaggle notebooks and discussions"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('KAGGLE_API_KEY')

    def extract_notebook(self, notebook_path: str) -> Dict[str, Any]:
        """Extract content from a Kaggle notebook (IPYNB file)"""
        if notebook_path.startswith('http') or '/' in notebook_path:
            return self._fetch_notebook_from_kaggle(notebook_path)
        else:
            return self._extract_local_notebook(notebook_path)

    def _extract_local_notebook(self, filepath: str) -> Dict[str, Any]:
        """Extract content from a local IPYNB file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        cells = []
        for i, cell in enumerate(notebook.get('cells', [])):
            cell_type = cell.get('cell_type', '')
            source = ''.join(cell.get('source', []))

            cells.append({
                'index': i,
                'type': cell_type,
                'content': source,
                'metadata': cell.get('metadata', {})
            })

        return {
            'metadata': notebook.get('metadata', {}),
            'cells': cells,
            'title': os.path.basename(filepath).replace('.ipynb', ''),
            'author': 'unknown',
            'date': datetime.now().isoformat()
        }

    def _fetch_notebook_from_kaggle(self, notebook_path: str) -> Dict[str, Any]:
        """Fetch notebook from Kaggle using API"""
        # Implementation for Kaggle API
        print(f"Fetching notebook {notebook_path} from Kaggle...")
        # This would use the Kaggle API
        return {}

    def extract_discussion(self, discussion_url: str) -> Dict[str, Any]:
        """Extract content from a Kaggle discussion"""
        # Mock implementation - in production, use Kaggle API or scraping
        return {
            'id': 'disc123',
            'title': 'Sample Discussion',
            'content': 'This is a long discussion about feature engineering. Many people share their experiences and code snippets.',
            'author': 'kaggle_user',
            'date': datetime.now().isoformat(),
            'comments': [
                {
                    'text': 'Great point! I also found that feature selection is crucial.',
                    'author': 'commenter1',
                    'date': datetime.now().isoformat(),
                    'level': 1
                },
                {
                    'text': 'Here is a code snippet that demonstrates this:\n\ndef select_features(X, y):\n    from sklearn.feature_selection import SelectKBest\n    selector = SelectKBest(k=10)\n    X_selected = selector.fit_transform(X, y)\n    return X_selected',
                    'author': 'commenter2',
                    'date': datetime.now().isoformat(),
                    'level': 1
                }
            ]
        }

    def search_kaggle_content(self, query: str, content_type: str = 'all', max_results: int = 100) -> List[Dict]:
        """
        Search for Kaggle content based on query
        """
        # This would use Kaggle's search API
        # Simplified version returning mock data
        results = []
        for i in range(max_results):
            results.append({
                'id': f'content_{i}',
                'title': f'Result {i} for {query}',
                'type': content_type if content_type != 'all' else ['notebook', 'discussion'][i % 2],
                'url': f'https://kaggle.com/...',
                'relevance_score': np.random.random()
            })
        return sorted(results, key=lambda x: x['relevance_score'], reverse=True)


class CodeAnalyzer:
    """Analyze and describe Python code from Kaggle notebooks"""
    def code_to_description(self, code: str) -> str:
        """Analyze code and return description and metadata"""
        analysis = {
            'description': '',
            'imports': [],
            'functions': [],
            'classes': [],
            'patterns': [],
        }

        try:
            tree = ast.parse(code)

            analysis['imports'] = self._extract_imports(tree)
            analysis['functions'] = self._extract_functions(tree)
            analysis['classes'] = self._extract_classes(tree)
            analysis['patterns'] = self._detect_patterns(code)
            analysis['description'] = self._generate_description(analysis)

        except SyntaxError:
            analysis['description'] = "Code snippet (syntax could not be parsed)"
        except:
            analysis['description'] = f"Code snippet"

        return analysis

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract imported modules"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return list(set(imports))

    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function definitions with descriptions"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get docstring
                docstring = ast.get_docstring(node)
                description = docstring.split('\n')[0] if docstring else f"function {node.name}"

                functions.append({
                    'name': node.name,
                    'description': description,
                    'args': [arg.arg for arg in node.args.args]
                })
        return functions

    def _extract_classes(self, tree: ast.AST) -> List[Dict]:
        """Extract class definitions"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                })
        return classes

    def _detect_patterns(self, code: str) -> List[str]:
        """Detect common ML/DL patterns in code"""
        patterns = []

        if 'sklearn' in code or 'scikit-learn' in code:
            patterns.append('scikit-learn')
        if 'tensorflow' in code or 'tf.' in code:
            patterns.append('TensorFlow')
        if 'torch' in code:
            patterns.append('PyTorch')
        if 'keras' in code:
            patterns.append('Keras')
        if 'plt.' in code or 'sns.' in code:
            patterns.append('visualization')
        if '.fit(' in code:
            patterns.append('model_training')
        if '.predict(' in code:
            patterns.append('prediction')
        if 'GridSearchCV' in code:
            patterns.append('hyperparameter_tuning')
        if 'train_test_split' in code:
            patterns.append('data_splitting')
        if 'cross_val_score' in code:
            patterns.append('cross_validation')

        return patterns

    def _generate_description(self, analysis: Dict) -> str:
        """Generate human-readable description from analysis"""
        parts = []

        if analysis['imports']:
            main_libs = [imp for imp in analysis['imports'] if imp in
                         ['tensorflow', 'torch', 'sklearn', 'pandas', 'numpy', 'matplotlib']]
            if main_libs:
                parts.append(f"Uses {', '.join(main_libs[:3])}")

        if analysis['functions']:
            parts.append(f"Defines {len(analysis['functions'])} functions")

        if analysis['classes']:
            parts.append(f"Defines {len(analysis['classes'])} classes")

        if analysis['patterns']:
            parts.append(f"Implements {', '.join(analysis['patterns'])}")

        return '. '.join(parts)


class LangChainChunker:
    """Use LangChain text splitters for intelligent chunking"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LangChain splitters
        self.python_splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )

        self.token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_code_cell(self, code: str, cell_index: int) -> List[Dict]:
        """
        Split a code cell using PythonCodeTextSplitter
        Also identify functions/classes for better metadata
        """
        chunks = []

        # First, try to identify functions and classes for better chunking
        try:
            tree = ast.parse(code)
            functions_and_classes = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else None

                    # Get the code for this node
                    lines = code.split('\n')
                    if end_line:
                        node_code = '\n'.join(lines[start_line - 1:end_line])
                    else:
                        # Approximate if end_line not available
                        node_code = ast.unparse(node)

                    functions_and_classes.append({
                        'type': 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class',
                        'name': node.name,
                        'code': node_code,
                        'start_line': start_line,
                        'end_line': end_line
                    })
        except:
            functions_and_classes = []

        # Use LangChain's Python splitter
        if functions_and_classes:
            # If we have identified functions, chunk by function
            for item in functions_and_classes:
                # Further split large functions if needed
                if len(item['code'].split('\n')) > self.chunk_size / 10:  # Rough estimate
                    sub_chunks = self.python_splitter.split_text(item['code'])
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunks.append({
                            'type': item['type'],
                            'name': item['name'],
                            'code': sub_chunk,
                            'description': f"{item['type']} {item['name']} (part {j + 1})",
                            'start_line': item['start_line'],
                            'is_sub_chunk': True,
                            'sub_index': j
                        })
                else:
                    chunks.append({
                        'type': item['type'],
                        'name': item['name'],
                        'code': item['code'],
                        'description': f"{item['type']} {item['name']}",
                        'start_line': item['start_line'],
                        'is_sub_chunk': False
                    })
        else:
            # No clear structure, use recursive splitting
            split_code = self.python_splitter.split_text(code)
            for i, chunk in enumerate(split_code):
                chunks.append({
                    'type': 'code_snippet',
                    'code': chunk,
                    'description': f"Code snippet (part {i + 1})",
                    'start_line': i * self.chunk_size + 1,
                    'is_sub_chunk': False
                })

        # Add cell index to all chunks
        for chunk in chunks:
            chunk['cell_index'] = cell_index

        return chunks

    def chunk_markdown_cell(self, markdown: str, cell_index: int) -> List[Dict]:
        """
        Split markdown using MarkdownTextSplitter
        """
        chunks = []

        # Use LangChain's markdown splitter
        split_markdown = self.markdown_splitter.split_text(markdown)

        for i, chunk in enumerate(split_markdown):
            # Try to extract header if present
            header_match = re.match(r'^#{1,6}\s+(.+)$', chunk.split('\n')[0], re.MULTILINE)
            header = header_match.group(1) if header_match else None

            chunks.append({
                'text': chunk,
                'header': header,
                'type': 'markdown_section',
                'cell_index': cell_index,
                'sub_index': i
            })

        return chunks

    def chunk_discussion(self, discussion_data: Dict) -> List[Dict]:
        """
        Split discussion content using recursive splitter
        """
        chunks = []

        # Main post
        main_post = discussion_data.get('content', '')
        if main_post:
            post_chunks = self.recursive_splitter.split_text(main_post)
            for i, chunk in enumerate(post_chunks):
                chunks.append({
                    'type': 'discussion_post',
                    'text': chunk,
                    'author': discussion_data.get('author'),
                    'date': discussion_data.get('date'),
                    'position': i,
                    'thread_level': 0,
                    'is_main_post': True
                })

        # Comments
        for i, comment in enumerate(discussion_data.get('comments', [])):
            comment_text = comment.get('text', '')
            if comment_text:
                # Check if comment contains code
                if 'def ' in comment_text or 'import ' in comment_text or '=' in comment_text:
                    # Use code splitter for code-heavy comments
                    comment_chunks = self.python_splitter.split_text(comment_text)
                    chunk_type = 'code_comment'
                else:
                    comment_chunks = self.recursive_splitter.split_text(comment_text)
                    chunk_type = 'comment'

                for j, chunk in enumerate(comment_chunks):
                    chunks.append({
                        'type': chunk_type,
                        'text': chunk,
                        'author': comment.get('author'),
                        'date': comment.get('date'),
                        'position': i,
                        'sub_position': j,
                        'thread_level': comment.get('level', 1),
                        'parent_id': comment.get('parent_id')
                    })

        return chunks


class TagGenerator:
    """Generate tags for Kaggle content based on predefined enum"""

    def __init__(self):
        self.tag_keywords = {
            KaggleTag.COMPUTER_VISION: ['cv2', 'opencv', 'image', 'cnn', 'conv', 'resnet', 'vgg', 'yolo'],
            KaggleTag.NLP: ['nlp', 'text', 'bert', 'transformer', 'token', 'sentence', 'word2vec', 'glove'],
            KaggleTag.TABULAR: ['pandas', 'dataframe', 'tabular', 'csv', 'excel', 'table'],
            KaggleTag.TIME_SERIES: ['time', 'series', 'date', 'temporal', 'seasonal', 'forecast'],

            KaggleTag.DEEP_LEARNING: ['neural', 'deep', 'layer', 'activation', 'backprop'],
            KaggleTag.MACHINE_LEARNING: ['machine learning', 'ml', 'algorithm', 'model'],
            KaggleTag.ENSEMBLE: ['ensemble', 'voting', 'stacking', 'bagging', 'boosting'],
            KaggleTag.FEATURE_ENGINEERING: ['feature', 'engineering', 'extraction', 'selection'],
            KaggleTag.EDA: ['eda', 'exploratory', 'analysis', 'visualization', 'distribution'],

            KaggleTag.PYTORCH: ['torch', 'pytorch', 'nn.Module'],
            KaggleTag.TENSORFLOW: ['tensorflow', 'tf.', 'keras'],
            KaggleTag.SKLEARN: ['sklearn', 'scikit'],
            KaggleTag.XGBOOST: ['xgboost', 'xgb'],
            KaggleTag.LIGHTGBM: ['lightgbm', 'lgb'],

            KaggleTag.TUTORIAL: ['tutorial', 'guide', 'introduction', 'basics'],
            KaggleTag.SOLUTION: ['solution', 'approach', 'method', 'implementation'],
            KaggleTag.DISCUSSION_TOPIC: ['discussion', 'topic', 'thread'],
            KaggleTag.QUESTION: ['question', 'help', 'issue', 'problem'],
            KaggleTag.ANSWER: ['answer', 'solution', 'fix'],

            KaggleTag.TEXT: ['text', 'string', 'character'],
            KaggleTag.NUMERIC: ['int', 'float', 'numeric', 'number'],

            KaggleTag.HIGH_SCORE: ['high score', 'top', 'leaderboard', 'winning'],
            KaggleTag.ENSEMBLE_SOLUTION: ['ensemble', 'blend', 'stack'],
            KaggleTag.BENCHMARK: ['benchmark', 'baseline', 'simple']
        }

    def generate_tags(self, text: str, code: Optional[str] = None,
                      chunk_type: Optional[ChunkType] = None,
                      metadata: Optional[Dict] = None) -> List[str]:
        """
        Generate tags based on content
        """
        tags = set()

        # Add chunk type as tag
        if chunk_type:
            tags.add(chunk_type.value)

        # Combine text for analysis
        full_text = text.lower()
        if code:
            full_text += ' ' + code.lower()
        if metadata:
            full_text += ' ' + json.dumps(metadata).lower()

        # Check each tag's keywords
        for tag, keywords in self.tag_keywords.items():
            for keyword in keywords:
                if keyword.lower() in full_text:
                    tags.add(tag.value)
                    break

        # Content-specific heuristics
        if 'def ' in full_text or 'class ' in full_text:
            tags.add('has_implementation')

        if 'plt.' in full_text or 'plot' in full_text or 'figure' in full_text:
            tags.add('visualization')

        if 'accuracy' in full_text or 'score' in full_text or 'metric' in full_text:
            tags.add('evaluation')

        if 'error' in full_text or 'bug' in full_text or 'issue' in full_text:
            tags.add('debugging')

        if 'install' in full_text or 'pip' in full_text or 'conda' in full_text:
            tags.add('installation')

        return list(tags)


class VectorStore:
    """Manage vector storage and retrieval of Kaggle content"""

    def __init__(self, persist_directory: str = "./kaggle_vector_store"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)

        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        except ValueError:
            return self.chroma_client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )

    def add_chunk(self, chunk: ContentChunk):
        """Add a single chunk to vector store"""
        # Prepare text for embedding
        text_to_embed = chunk.text
        if chunk.code:
            text_to_embed += f"\n\nCode:\n{chunk.code}"
        if chunk.code_description:
            text_to_embed += f"\n\nDescription: {chunk.code_description}"

        # Generate embedding
        embedding = self.embedding_model.encode(text_to_embed[:5000]).tolist()  # Limit length

        # Prepare metadata (ensure all values are strings or numbers)
        metadata = {
            "chunk_id": chunk.id,
            "source_id": chunk.source_id,
            "source_title": str(chunk.source_title)[:100],
            "source_url": str(chunk.source_url)[:200],
            "chunk_type": chunk.chunk_type.value,
            "content_type": chunk.content_type.value,
            "tags": json.dumps(chunk.tags)[:1000],
            "position": chunk.position,
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
            documents=[text_to_embed[:1000]],  # Store preview
            metadatas=[metadata],
            ids=[chunk.id]
        )

    def search_chunks(self, query: str, content_type: Optional[ContentType] = None,
                      chunk_type: Optional[ChunkType] = None,
                      tags: Optional[List[str]] = None,
                      n_results: int = 10) -> List[Dict]:
        """
        Search for relevant chunks
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Build filter
        where_filter = {}
        if content_type:
            where_filter["content_type"] = content_type.value
        if chunk_type:
            where_filter["chunk_type"] = chunk_type.value

        # Search
        results = self.chunks_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results * 2,  # Get more for filtering
            where=where_filter if where_filter else None
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for idx, (chunk_id, metadata, distance) in enumerate(zip(
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


class KaggleRAGPipeline:
    """Main pipeline for processing Kaggle content and building RAG system"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.extractor = KaggleExtractor()
        self.chunker = LangChainChunker(chunk_size, chunk_overlap)
        self.code_analyzer = CodeAnalyzer()
        self.tag_generator = TagGenerator()
        self.vector_store = VectorStore()

    def process_notebook(self, notebook_source: str, source_type: str = 'local') -> KaggleSource:
        """
        Process a Kaggle notebook and chunk it
        """
        # Extract notebook content
        notebook_data = self.extractor.extract_notebook(notebook_source)

        # Create source record
        source_id = str(uuid.uuid4())
        source = KaggleSource(
            id=source_id,
            title=notebook_data.get('title', 'Untitled'),
            url=f"https://kaggle.com/{notebook_source}" if source_type == 'kaggle' else '',
            content_type=ContentType.NOTEBOOK,
            author=notebook_data.get('author', 'unknown'),
            date=notebook_data.get('date', datetime.now().isoformat()),
            metadata={
                'cell_count': len(notebook_data.get('cells', [])),
                'source_type': source_type
            }
        )

        # Process each cell
        chunks = []
        for cell in notebook_data.get('cells', []):
            cell_type = cell['type']
            content = cell['content']
            cell_index = cell['index']

            if cell_type == 'markdown':
                # Chunk markdown using LangChain
                markdown_chunks = self.chunker.chunk_markdown_cell(content, cell_index)

                for i, chunk_data in enumerate(markdown_chunks):
                    chunk_id = str(uuid.uuid4())

                    # Analyze text for tags
                    tags = self.tag_generator.generate_tags(
                        text=chunk_data['text'],
                        chunk_type=ChunkType.MARKDOWN_CELL,
                        metadata={'header': chunk_data.get('header')}
                    )

                    # Add specific tag if it's a header
                    if chunk_data.get('header'):
                        tags.append('section_header')

                    chunk = ContentChunk(
                        id=chunk_id,
                        source_id=source_id,
                        source_title=source.title,
                        source_url=source.url,
                        chunk_type=ChunkType.MARKDOWN_CELL,
                        content_type=ContentType.NOTEBOOK,
                        text=chunk_data['text'],
                        tags=tags,
                        metadata={
                            'header': chunk_data.get('header', ''),
                            'cell_index': cell_index,
                            'sub_index': i,
                            'is_header': bool(chunk_data.get('header'))
                        },
                        position=cell_index * 1000 + i
                    )

                    chunks.append(chunk)

            elif cell_type == 'code':
                # Analyze code for description
                analysis = self.code_analyzer.analyze_code(content)

                # Chunk code using LangChain
                code_chunks = self.chunker.chunk_code_cell(content, cell_index)

                for i, chunk_data in enumerate(code_chunks):
                    chunk_id = str(uuid.uuid4())

                    # Determine chunk type
                    if chunk_data['type'] in ['function', 'class']:
                        chunk_type = ChunkType.FUNCTION if chunk_data['type'] == 'function' else ChunkType.CLASS
                    else:
                        chunk_type = ChunkType.CODE_SNIPPET

                    # Generate tags
                    tags = self.tag_generator.generate_tags(
                        text=chunk_data['description'],
                        code=chunk_data['code'],
                        chunk_type=chunk_type,
                        metadata=analysis
                    )

                    chunk = ContentChunk(
                        id=chunk_id,
                        source_id=source_id,
                        source_title=source.title,
                        source_url=source.url,
                        chunk_type=chunk_type,
                        content_type=ContentType.CODE,
                        text=chunk_data['description'],
                        code=chunk_data['code'],
                        code_description=chunk_data['description'],
                        tags=tags,
                        metadata={
                            'code_type': chunk_data['type'],
                            'name': chunk_data.get('name', ''),
                            'start_line': chunk_data.get('start_line', 0),
                            'cell_index': cell_index,
                            'sub_index': i,
                            'is_sub_chunk': chunk_data.get('is_sub_chunk', False),
                            **analysis
                        },
                        position=cell_index * 1000 + i + 500  # Code after markdown
                    )

                    chunks.append(chunk)

        # Add all chunks to vector store
        for chunk in chunks:
            self.vector_store.add_chunk(chunk)
            source.chunks.append(chunk.id)

        print(f"Processed notebook '{source.title}' into {len(chunks)} chunks")
        return source

    def process_discussion(self, discussion_url: str) -> KaggleSource:
        """
        Process a Kaggle discussion and chunk it
        """
        # Extract discussion content
        discussion_data = self.extractor.extract_discussion(discussion_url)

        # Create source record
        source_id = str(uuid.uuid4())
        source = KaggleSource(
            id=source_id,
            title=discussion_data.get('title', 'Untitled Discussion'),
            url=discussion_url,
            content_type=ContentType.DISCUSSION,
            author=discussion_data.get('author', 'unknown'),
            date=discussion_data.get('date', datetime.now().isoformat()),
            metadata={
                'comment_count': len(discussion_data.get('comments', []))
            }
        )

        # Chunk discussion using LangChain
        discussion_chunks = self.chunker.chunk_discussion(discussion_data)

        chunks = []
        for i, chunk_data in enumerate(discussion_chunks):
            chunk_id = str(uuid.uuid4())

            # Determine chunk type
            if chunk_data['type'] == 'discussion_post':
                chunk_type = ChunkType.DISCUSSION_POST
            elif chunk_data['type'] == 'code_comment':
                chunk_type = ChunkType.CODE_SNIPPET
            else:
                chunk_type = ChunkType.COMMENT

            # Check if it's a question or answer
            text_lower = chunk_data['text'].lower()
            if '?' in chunk_data['text'] or any(q in text_lower for q in ['how to', 'why', 'help']):
                tags = [KaggleTag.QUESTION.value]
            elif any(a in text_lower for a in ['answer', 'solution', 'try this']):
                tags = [KaggleTag.ANSWER.value]
            else:
                tags = []

            # Generate additional tags
            more_tags = self.tag_generator.generate_tags(
                text=chunk_data['text'],
                chunk_type=chunk_type
            )
            tags.extend(more_tags)

            chunk = ContentChunk(
                id=chunk_id,
                source_id=source_id,
                source_title=source.title,
                source_url=source.url,
                chunk_type=chunk_type,
                content_type=ContentType.DISCUSSION,
                text=chunk_data['text'],
                tags=tags,
                metadata={
                    'author': chunk_data.get('author'),
                    'date': chunk_data.get('date'),
                    'thread_level': chunk_data.get('thread_level', 0),
                    'position': chunk_data.get('position', i),
                    'sub_position': chunk_data.get('sub_position', 0),
                    'is_main_post': chunk_data.get('is_main_post', False),
                    'parent_id': chunk_data.get('parent_id')
                },
                position=i
            )

            chunks.append(chunk)

        # Add all chunks to vector store
        for chunk in chunks:
            self.vector_store.add_chunk(chunk)
            source.chunks.append(chunk.id)

        print(f"Processed discussion '{source.title}' into {len(chunks)} chunks")
        return source

    def batch_process(self, sources: List[Dict]) -> List[KaggleSource]:
        """
        Process multiple sources (both notebooks and discussions)
        """
        processed_sources = []

        for source_info in tqdm(sources, desc="Processing Kaggle content"):
            try:
                if source_info['type'] == 'notebook':
                    source = self.process_notebook(
                        source_info['source'],
                        source_info.get('source_type', 'local')
                    )
                elif source_info['type'] == 'discussion':
                    source = self.process_discussion(source_info['source'])
                else:
                    print(f"Unknown source type: {source_info['type']}")
                    continue

                processed_sources.append(source)

            except Exception as e:
                print(f"Error processing {source_info}: {str(e)}")
                continue

        return processed_sources

    def search(self, query: str, content_type: Optional[ContentType] = None,
               chunk_type: Optional[ChunkType] = None,
               tags: Optional[List[str]] = None,
               n_results: int = 10) -> List[Dict]:
        """
        Search for relevant chunks
        """
        return self.vector_store.search_chunks(
            query=query,
            content_type=content_type,
            chunk_type=chunk_type,
            tags=tags,
            n_results=n_results
        )

    def get_context_for_query(self, query: str, n_chunks: int = 5) -> str:
        """
        Get relevant chunks and format them as context for an LLM
        """
        results = self.search(query, n_results=n_chunks)

        context_parts = []
        for i, result in enumerate(results):
            # Get full chunk text (would need to retrieve from storage)
            # For now, use what we have
            context_parts.append(f"[{i + 1}] From: {result['source_title']}")
            context_parts.append(f"Type: {result['chunk_type']}")
            context_parts.append(f"Tags: {', '.join(result['tags'][:5])}")
            context_parts.append("---")

        return '\n'.join(context_parts)


# Example usage and testing
def main():
    """Example of how to use the Kaggle RAG system"""

    # Initialize pipeline
    pipeline = KaggleRAGPipeline()

    # Example 1: Process a local notebook
    print("Processing local notebook...")
    notebook_content = pipeline.process_notebook(
        "sample_notebook.ipynb",
        source_type='local'
    )
    print(f"Processed notebook: {notebook_content.title}")
    print(f"Tags: {notebook_content.tags}")
    print(f"Code description: {notebook_content.code_description[:200]}...")

    # Example 2: Process a discussion (simulated)
    print("\nProcessing discussion...")
    discussion_content = pipeline.process_discussion(
        "https://kaggle.com/discussion/12345"
    )
    print(f"Processed discussion: {discussion_content.title}")
    print(f"Tags: {discussion_content.tags}")

    # Example 3: Batch process multiple sources
    sources = [
        {'type': 'notebook', 'source': 'user1/notebook1.ipynb', 'source_type': 'local'},
        {'type': 'notebook', 'source': 'user2/notebook2.ipynb', 'source_type': 'local'},
        {'type': 'discussion', 'source': 'https://kaggle.com/discussion/12346'},
        {'type': 'discussion', 'source': 'https://kaggle.com/discussion/12347'},
    ]

    print("\nBatch processing...")
    processed = pipeline.batch_process(sources)
    print(f"Processed {len(processed)} items")

    # Example 4: Search
    print("\nSearching for content...")
    results = pipeline.search(
        query="How to do feature engineering for tabular data?",
        content_type=ContentType.DISCUSSION,
        tags=['feature_engineering', 'tabular'],
        n_results=5
    )

    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i + 1}. Type: {result['type']}")
        print(f"   Title: {result['metadata'].get('title', 'Unknown')}")
        print(f"   Similarity: {result['similarity_score']:.3f}")
        print(f"   Tags: {result['metadata'].get('tags', '[]')[:100]}...")

    # Example 5: Agent-based querying with specific requirements
    print("\n=== Agent Query Examples ===")

    # Agent 1: Looking for PyTorch code solutions
    code_results = pipeline.search(
        query="image classification using PyTorch",
        content_type=ContentType.CODE,
        tags=['computer_vision', 'pytorch', 'deep_learning'],
        n_results=3
    )

    print("\nAgent 1 (Code Specialist) Results:")
    for r in code_results:
        print(f"- {r['metadata']['title']} (Score: {r['similarity_score']:.2f})")

    # Agent 2: Looking for discussion insights
    discussion_results = pipeline.search(
        query="best practices for handling missing values",
        content_type=ContentType.DISCUSSION,
        tags=['tabular', 'feature_engineering'],
        n_results=3
    )

    print("\nAgent 2 (Discussion Analyst) Results:")
    for r in discussion_results:
        print(f"- {r['metadata']['title']} (Score: {r['similarity_score']:.2f})")


# Utility functions for production use
def create_kaggle_content_index(
        notebook_dirs: List[str],
        discussion_urls: List[str],
        output_dir: str = "./kaggle_index"
) -> KaggleRAGPipeline:
    """
    Create a complete index from multiple Kaggle sources
    """
    pipeline = KaggleRAGPipeline()

    # Process notebooks from directories
    sources = []

    for notebook_dir in notebook_dirs:
        for root, dirs, files in os.walk(notebook_dir):
            for file in files:
                if file.endswith('.ipynb'):
                    sources.append({
                        'type': 'notebook',
                        'source': os.path.join(root, file),
                        'source_type': 'local'
                    })

    for url in discussion_urls:
        sources.append({
            'type': 'discussion',
            'source': url,
            'source_type': 'kaggle'
        })

    # Batch process all sources
    pipeline.batch_process(sources)

    return pipeline

if __name__ == "__main__":
    main()