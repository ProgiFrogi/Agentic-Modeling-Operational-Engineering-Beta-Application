import json
from typing import List, Dict, Optional
from openai import OpenAI

import ast

# For text processing
import re
from langchain_text_splitters import (
    PythonCodeTextSplitter,
    MarkdownTextSplitter,
)

from rag.rag_types import ChunkTags


class CodeAnalyzer:
    """Analyze and describe Python code from Kaggle notebooks"""

    def __init__(self, llm_client: OpenAI, model_name: str, description_limit: int = 500):
        self._llm_client = llm_client
        self.model_name = model_name
        self.description_limit = description_limit

    def analyze_code(self, code: str) -> Dict:
        """Analyze code and return description and metadata"""
        analysis = {'description': self._generate_description(code), 'imports': [], 'functions': [], 'classes': []}

        try:
            tree = ast.parse(code)

            analysis['imports'] = self._extract_imports(tree)
            analysis['functions'] = self._extract_functions(tree)
            analysis['classes'] = self._extract_classes(tree)
        except Exception:
            pass

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

    def _generate_description(self, code: str) -> str:
        """Generate human-readable description using an LLM if available, otherwise fallback."""
        sys_prompt = (
            "You are a senior Python ML engineer. Summarize the given code analysis into one or two concise sentences. "
            "Respond with plain text explanations only, never code and never markdown. "
            "Avoid hedging, markdown and comments. "
            "Under no conditions continue given code. "
        )

        try:
            resp = self._llm_client.chat.completions.create(model=self.model_name, messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Here is the code: {code} \nSummarize in up to three sentences."}
            ], temperature=0.2, max_tokens=self.description_limit)
            msg = resp.choices[0].message.content.replace('<|im_start|>', '').strip()
            return msg
        except Exception:
            return code


class LangChainChunker:
    """Use LangChain text splitters for intelligent chunking"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.python_splitter = PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.markdown_splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_code_cell(self, code: str, cell_index: int) -> List[Dict]:
        """
        Split a code cell using PythonCodeTextSplitter
        """
        chunks = []

        split_code = self.python_splitter.split_text(code)
        for i, chunk in enumerate(split_code):
            chunks.append({
                'text': chunk,
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
            # Domains
            ChunkTags.COMPUTER_VISION: ['cv2', 'opencv', 'image', 'cnn', 'conv', 'resnet', 'vgg', 'yolo'],
            ChunkTags.NLP: ['nlp', 'text', 'bert', 'transformer', 'token', 'sentence', 'word2vec', 'glove'],
            ChunkTags.TABULAR: ['pandas', 'dataframe', 'tabular', 'csv', 'excel', 'table'],
            ChunkTags.TIME_SERIES: ['time series', 'timestamp', 'datetime', 'seasonal', 'forecast', 'arima', 'prophet'],

            # Techniques
            ChunkTags.DEEP_LEARNING: ['neural', 'deep learning', 'activation', 'backprop', 'dropout', 'optimizer'],
            ChunkTags.MACHINE_LEARNING: ['machine learning', 'ml algorithm', 'model'],
            ChunkTags.ENSEMBLE: ['ensemble', 'voting', 'stacking', 'bagging', 'boosting'],
            ChunkTags.FEATURE_ENGINEERING: ['feature engineering', 'feature extraction', 'feature selection'],
            ChunkTags.EDA: ['eda', 'exploratory data analysis', 'visualization', 'distribution', 'histogram'],
            ChunkTags.CROSS_VALIDATION: ['cross validation', 'cross_val_score', 'kfold', 'stratifiedkfold'],
            ChunkTags.HYPERPARAMETERS: ['hyperparameter', 'gridsearchcv', 'randomizedsearchcv', 'optuna', 'bayes'],

            # Tools
            ChunkTags.PYTORCH: ['torch', 'pytorch', 'nn.module'],
            ChunkTags.TENSORFLOW: ['tensorflow', 'tf.', 'keras'],
            ChunkTags.SKLEARN: ['sklearn', 'scikit-learn'],
            ChunkTags.XGBOOST: ['xgboost', 'xgb'],
            ChunkTags.LIGHTGBM: ['lightgbm', 'lgb'],
            ChunkTags.KERAS: ['keras'],

            # Content style
            ChunkTags.TUTORIAL: ['tutorial', 'guide', 'introduction', 'how to'],
            ChunkTags.SOLUTION: ['solution', 'approach', 'method', 'implementation'],
            ChunkTags.DISCUSSION_TOPIC: ['discussion', 'topic', 'thread'],
            ChunkTags.QUESTION: ['question', 'how do', 'help', 'issue', 'why'],
            ChunkTags.ANSWER: ['answer', 'try this', 'fix'],

            # Performance
            ChunkTags.HIGH_SCORE: ['high score', 'top', 'leaderboard', 'winning'],
            ChunkTags.ENSEMBLE_SOLUTION: ['ensemble', 'blend', 'stack'],
            ChunkTags.BENCHMARK: ['benchmark', 'baseline', 'simple'],

            # Code specific
            ChunkTags.INSTALLATION: ['pip install', 'conda install', 'pip3 install', 'apt-get install'],
            ChunkTags.IMPORTS: ['import '],
            ChunkTags.IMPLEMENTATION: ['def ', 'class '],
            ChunkTags.EVALUATION: ['accuracy', 'auc', 'f1', 'logloss', 'roc', 'metric', 'score'],
            ChunkTags.MODEL_TRAINING: ['.fit(', 'fit(', 'Trainer(', 'train('],
            ChunkTags.MODEL_PREDICTION: ['.predict(', 'predict(', 'inference'],
            ChunkTags.SPLIT: ['train_test_split', 'validation set', 'holdout'],
        }

    def generate_tags(self, text: str, code: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> List[ChunkTags]:
        """
        Generate tags based on content
        """
        tags = set()

        # Combine text for analysis
        full_text = (text or '').lower()
        if code:
            full_text += ' ' + code.lower()
        if metadata:
            try:
                full_text += ' ' + json.dumps(metadata).lower()
            except Exception:
                pass

        # Match tag keywords
        for tag, keywords in self.tag_keywords.items():
            for keyword in keywords:
                if keyword in full_text:
                    tags.add(tag)
                    break

        return list(tags)
