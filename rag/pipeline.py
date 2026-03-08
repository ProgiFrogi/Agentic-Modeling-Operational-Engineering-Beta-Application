import traceback
import uuid
from typing import List, Dict, Optional

from openai import OpenAI

from rag.chunk_work import LangChainChunker, CodeAnalyzer, TagGenerator
from rag.extractor import KaggleExtractor
from rag.storage import VectorStore
from rag.rag_types import ContentType, ChunkType, ContentChunk, KaggleSource
# Kaggle helpers
from tools.kaggle_utils import (
    search_competitions,
    search_kernels,
    download_kernel_notebook,
)

class KaggleRAGPipeline:
    """Main pipeline for processing Kaggle content and building RAG system"""

    def __init__(self, code_describe_llm: OpenAI, code_describe_model: str, chunk_size: int = 500, chunk_overlap: int = 50, min_chunk_length: int = 30):
        self.min_text_length = min_chunk_length

        self.extractor = KaggleExtractor()
        self.chunker = LangChainChunker(chunk_size, chunk_overlap)
        self.code_analyzer = CodeAnalyzer(code_describe_llm, code_describe_model)
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
                    if len(chunk_data['text']) < self.min_text_length:
                        continue
                    chunk_id = str(uuid.uuid4())

                    # Analyze text for tags
                    tags = self.tag_generator.generate_tags(
                        text=chunk_data['text'],
                        metadata={'header': chunk_data.get('header')}
                    )

                    chunk = ContentChunk(
                        id=chunk_id,
                        source_title=notebook_data['title'],
                        content_type=ContentType.NOTEBOOK,
                        chunk_type=ChunkType.MARKDOWN_CELL,
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
                        chunk_type = ChunkType.CLASS
                    elif analysis['functions']:
                        chunk_type = ChunkType.FUNCTION
                    else:
                        chunk_type = ChunkType.CODE_SNIPPET

                    tags = self.tag_generator.generate_tags(text=analysis['description'], code=chunk_data['code'])

                    chunk = ContentChunk(
                        id=chunk_id,
                        source_title=notebook_data['title'],
                        chunk_type=chunk_type,
                        content_type=ContentType.NOTEBOOK,
                        text=chunk_data['code'],
                        code=chunk_data['code'],
                        code_description=analysis['description'],
                        tags=tags,
                        metadata={
                            'cell_index': cell_index,
                            'sub_index': i,
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
                                download_dir: str = "./kaggle_notebooks"):
        """
        Use Kaggle search utilities to collect competitions, then fetch notebooks and discussions
        and process them into the vector store.
        """

        comps = search_competitions(query=query, max_results=n_competitions)
        if not comps:
            print("No competitions found for given query.")
            return

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
                    src = self.process_notebook(nb_path)
                    for chunk in src.chunks:
                        self.vector_store.add_chunk(chunk)
                except Exception:
                    print(f"Error processing notebook {ref}:")
                    traceback.print_exc()

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
