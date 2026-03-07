"""
Kaggle RAG System for Notebooks and Discussions
This system extracts, processes, and indexes Kaggle content for semantic search.
"""

import os
import json
import uuid
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# For code understanding
import ast

# For embeddings and vector storage
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Kaggle helpers
from tools.kaggle_utils import (
    search_competitions,
    search_kernels,
    download_kernel_notebook,
)

# For text processing
import re
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
    TokenTextSplitter
)


class ContentType(Enum):
    """Enum for content type classification"""
    NOTEBOOK = "notebook"
    DISCUSSION = "discussion"  # unsupported


class ChunkType(Enum):
    """Enum for chunk types within content"""
    MARKDOWN_CELL = "markdown_cell"
    DISCUSSION_POST = "discussion_post"
    COMMENT = "comment"
    EXPLANATION = "explanation"
    QUESTION = "question"
    ANSWER = "answer"
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


tags_all = [tag for tag in ChunkTags]
tags_all_str = [tag.value for tag in ChunkTags]


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
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_size: int = 0


@dataclass
class KaggleSource:
    """Represents the original source (notebook or discussion)"""
    id: str
    title: str
    content_type: ContentType
    chunks: List[ContentChunk] = field(default_factory=list)


class KaggleExtractor:
    """Extract content from Kaggle notebooks and discussions"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('KAGGLE_API_KEY')

    def extract_notebook(self, notebook_path: str) -> Dict[str, Any]:
        """Extract content from a local Kaggle notebook (IPYNB file)"""
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
        }


class CodeAnalyzer:
    """Analyze and describe Python code from Kaggle notebooks"""

    def analyze_code(self, code: str) -> Dict:
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
            analysis['description'] = self._generate_description(analysis)

        except SyntaxError:
            analysis['description'] = "Code snippet (syntax could not be parsed)"
        except Exception:
            analysis['description'] = "Code snippet"

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

    def chunk_code_cell(self, code: str, cell_index: int) -> List[Dict]:
        """
        Split a code cell using PythonCodeTextSplitter
        """
        chunks = []

        split_code = self.python_splitter.split_text(code)
        for i, chunk in enumerate(split_code):
            chunks.append({
                'code': chunk,
                'cell_index': cell_index,
            })

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
                'cell_index': cell_index,
            })

        return chunks


class TagGenerator:
    """Generate tags for Kaggle content based on predefined enum"""

    def __init__(self):
        self.tag_keywords = {
            ChunkTags.COMPUTER_VISION: ['cv2', 'opencv', 'image', 'cnn', 'conv', 'resnet', 'vgg', 'yolo'],
            ChunkTags.NLP: ['nlp', 'text', 'bert', 'transformer', 'token', 'sentence', 'word2vec', 'glove'],
            ChunkTags.TABULAR: ['pandas', 'dataframe', 'tabular', 'csv', 'excel', 'table'],
            ChunkTags.TIME_SERIES: ['time', 'series', 'date', 'temporal', 'seasonal', 'forecast'],

            ChunkTags.DEEP_LEARNING: ['neural', 'deep', 'layer', 'activation', 'backprop'],
            ChunkTags.MACHINE_LEARNING: ['machine learning', 'ml', 'algorithm', 'model'],
            ChunkTags.ENSEMBLE: ['ensemble', 'voting', 'stacking', 'bagging', 'boosting'],
            ChunkTags.FEATURE_ENGINEERING: ['feature', 'engineering', 'extraction', 'selection'],
            ChunkTags.EDA: ['eda', 'exploratory', 'analysis', 'visualization', 'distribution'],

            ChunkTags.PYTORCH: ['torch', 'pytorch', 'nn.Module'],
            ChunkTags.TENSORFLOW: ['tensorflow', 'tf.', 'keras'],
            ChunkTags.SKLEARN: ['sklearn', 'scikit'],
            ChunkTags.XGBOOST: ['xgboost', 'xgb'],
            ChunkTags.LIGHTGBM: ['lightgbm', 'lgb'],

            ChunkTags.TUTORIAL: ['tutorial', 'guide', 'introduction', 'basics'],
            ChunkTags.SOLUTION: ['solution', 'approach', 'method', 'implementation'],
            ChunkTags.DISCUSSION_TOPIC: ['discussion', 'topic', 'thread'],
            ChunkTags.QUESTION: ['question', 'help', 'issue', 'problem'],
            ChunkTags.ANSWER: ['answer', 'solution', 'fix'],

            ChunkTags.HIGH_SCORE: ['high score', 'top', 'leaderboard', 'winning'],
            ChunkTags.ENSEMBLE_SOLUTION: ['ensemble', 'blend', 'stack'],
            ChunkTags.BENCHMARK: ['benchmark', 'baseline', 'simple'],

            ChunkTags.INSTALLATION: ['conda', 'install', 'pip'],
            ChunkTags.IMPORTS: ['import'],
            ChunkTags.IMPLEMENTATION: ['def', 'class'],
            ChunkTags.EVALUATION: ['accuracy', 'score', 'metric'],
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

        return list(tags)


class VectorStore:
    """Manage vector storage and retrieval of Kaggle content"""

    def __init__(self, persist_directory: str = "./kaggle_vector_store", encode_limit: int = 5000,
                 preview_limit: int = 1000):
        self.encode_limit = encode_limit
        self.preview_limit = preview_limit

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
        if chunk.code_description:
            text_to_embed += chunk.code_description
        else:
            text_to_embed = chunk.text

        # Generate embedding
        embedding = self.embedding_model.encode(text_to_embed[:self.encode_limit]).tolist()

        # Prepare metadata (ensure all values are strings or numbers)
        metadata = {
            "chunk_id": chunk.id,
            "source_title": str(chunk.source_title)[:100],
            "chunk_type": chunk.chunk_type.value,
            "content_type": chunk.content_type.value,
            "tags": json.dumps(chunk.tags)[:1000],
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
            documents=[text_to_embed[:self.preview_limit]],
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

    def process_notebook(self, notebook_source: str) -> KaggleSource:
        """
        Process a Kaggle notebook and chunk it
        """
        # Extract notebook content
        notebook_data = self.extractor.extract_notebook(notebook_source)

        # Process each cell
        chunks = []
        for cell in notebook_data.get('cells', []):
            cell_type = cell['type']
            content = cell['content']
            cell_index = cell['index']

            if cell_type == 'markdown':
                # Chunk markdown
                markdown_chunks = self.chunker.chunk_markdown_cell(content, cell_index)

                for i, chunk_data in enumerate(markdown_chunks):
                    chunk_id = str(uuid.uuid4())

                    # Analyze text for tags
                    tags = self.tag_generator.generate_tags(
                        text=chunk_data['text'],
                        chunk_type=ChunkType.MARKDOWN_CELL,
                        metadata={'header': chunk_data.get('header')}
                    )

                    chunk = ContentChunk(
                        id=chunk_id,
                        source_title=notebook_data['title'],
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
                        chunk_size=len(chunk_data['text'])
                    )

                    chunks.append(chunk)

            elif cell_type == 'code':
                # Chunk code
                code_chunks = self.chunker.chunk_code_cell(content, cell_index)

                for i, chunk_data in enumerate(code_chunks):
                    chunk_id = str(uuid.uuid4())
                    analysis = self.code_analyzer.analyze_code(chunk_data['code'])

                    # Determine chunk type
                    if analysis['classes']:
                        chunk_type = ChunkType.FUNCTION
                    elif analysis['functions']:
                        chunk_type = ChunkType.CLASS
                    else:
                        chunk_type = ChunkType.CODE_SNIPPET

                    # Generate tags
                    tags = self.tag_generator.generate_tags(
                        text=analysis['description'],
                        code=chunk_data['code'],
                        chunk_type=chunk_type,
                        metadata=analysis
                    )

                    chunk = ContentChunk(
                        id=chunk_id,
                        source_title=notebook_data['title'],
                        chunk_type=chunk_type,
                        content_type=ContentType.NOTEBOOK,
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
                    )

                    chunks.append(chunk)

        source = KaggleSource(
            id=str(uuid.uuid4()),
            title=notebook_data['title'],
            content_type=ContentType.NOTEBOOK,
            chunks=chunks
        )

        print(f"Processed notebook '{source.title}' into {len(chunks)} chunks")
        return source

    def build_index_from_kaggle(self, query: Optional[str] = None, n_competitions: int = 3,
                                notebooks_per_comp: int = 5, discussions_per_comp: int = 5,
                                download_dir: str = "./kaggle_notebooks") -> List[KaggleSource]:
        """
        Use Kaggle search utilities to collect competitions, then fetch notebooks and discussions
        and process them into the vector store.
        """

        sources: List[KaggleSource] = []

        comps = search_competitions(query=query, max_results=n_competitions)
        if not comps:
            print("No competitions found for given query.")
            return sources

        for comp in comps:
            comp_ref = comp.get('ref')
            comp_title = comp.get('title')
            print(f"Processing competition: {comp_title} ({comp_ref})")

            # Notebooks for this competition
            kernels = search_kernels(competition=comp_ref, max_results=notebooks_per_comp)
            for k in kernels:
                ref = k.get('ref')
                nb_path = download_kernel_notebook(ref, path=download_dir)
                if not nb_path:
                    continue
                try:
                    src = self.process_notebook(nb_path, source_type='local')
                    sources.append(src)
                except Exception as e:
                    print(f"Error processing notebook {ref}: {e}")

        return sources

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


# Example usage
def main():
    """Build an index from Kaggle and demonstrate a search."""
    pipeline = KaggleRAGPipeline()

    try:
        print("Building index from Kaggle...")
        sources = pipeline.build_index_from_kaggle(
            query="tabular",
            n_competitions=1,
            notebooks_per_comp=1,
        )
        print(f"Indexed {len(sources)} sources from Kaggle.")
    except Exception as e:
        print(f"Kaggle indexing skipped: {e}")

    print("\nSearch demo:")
    results = pipeline.search(
        query="tabular feature engineering",
        n_results=5,
    )
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['chunk_type']}] {r['source_title']} (score={r['similarity_score']:.3f})")


if __name__ == "__main__":
    main()
