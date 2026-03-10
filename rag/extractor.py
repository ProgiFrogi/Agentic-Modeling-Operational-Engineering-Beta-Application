import os
import nbformat
from typing import Dict, Any, Optional
import re

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
            notebook = nbformat.read(f, nbformat.NO_CONVERT)

        cells = []
        for i, cell in enumerate(notebook.cells):
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