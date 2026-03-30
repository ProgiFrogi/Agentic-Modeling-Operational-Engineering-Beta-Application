import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import pandas as pd


class SessionManager:
    """Управляет сессионной папкой для работы с данными"""

    def __init__(self, source_dir: str = "data", session_dir: Optional[str] = None):
        self.source_dir = Path(source_dir)

        # Создаем уникальную сессию
        if session_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = Path(f"sessions/session_{timestamp}")
        else:
            self.session_dir = Path(session_dir)

        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Копируем исходные файлы
        self._copy_initial_files()

        # Пути к рабочим файлам
        self.train_path = self.session_dir / "train.csv"
        self.test_path = self.session_dir / "test.csv"

        # Создаем папку для метаданных
        self.metadata_dir = self.session_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def __getstate__(self):
        """Для безопасной сериализации - возвращаем только строковые пути"""
        return {
            "source_dir": str(self.source_dir),
            "session_dir": str(self.session_dir),
            "train_path": str(self.train_path),
            "test_path": str(self.test_path),
            "metadata_dir": str(self.metadata_dir)
        }

    def __setstate__(self, state):
        """Восстанавливаем объект из сериализованного состояния"""
        self.source_dir = Path(state["source_dir"])
        self.session_dir = Path(state["session_dir"])
        self.train_path = Path(state["train_path"])
        self.test_path = Path(state["test_path"])
        self.metadata_dir = Path(state["metadata_dir"])

    def _copy_initial_files(self):
        """Копирует исходные файлы в сессионную папку"""
        if self.source_dir.exists():
            for file in self.source_dir.glob("*.csv"):
                shutil.copy2(file, self.session_dir / file.name)
            # Также копируем txt файлы
            for file in self.source_dir.glob("*.txt"):
                shutil.copy2(file, self.session_dir / file.name)
            print(f"Copied files from {self.source_dir} to {self.session_dir}")

    def get_train_path(self) -> Path:
        """Возвращает путь к train.csv"""
        return self.train_path

    def get_test_path(self) -> Path:
        """Возвращает путь к test.csv"""
        return self.test_path

    def get_data_dir(self) -> Path:
        """Возвращает путь к сессионной папке"""
        return self.session_dir

    def save_dataframe(self, df: pd.DataFrame, name: str) -> Path:
        """Сохраняет датафрейм в сессионную папку"""
        file_path = self.session_dir / f"{name}.csv"
        df.to_csv(file_path, index=False)
        return file_path

    def load_dataframe(self, name: str) -> Optional[pd.DataFrame]:
        """Загружает датафрейм из сессионной папки"""
        file_path = self.session_dir / f"{name}.csv"
        if file_path.exists():
            return pd.read_csv(file_path)
        return None

    def save_metadata(self, key: str, data: Dict[str, Any]) -> Path:
        """Сохраняет метаданные в JSON файл"""
        metadata_path = self.metadata_dir / f"{key}.json"
        with open(metadata_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return metadata_path

    def load_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Загружает метаданные из JSON файла"""
        metadata_path = self.metadata_dir / f"{key}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None

    def list_files(self) -> list:
        """Список файлов в сессионной папке"""
        return [f.name for f in self.session_dir.glob("*") if f.is_file()]

    def cleanup(self):
        """Очистка сессионной папки (опционально)"""
        try:
            shutil.rmtree(self.session_dir)
            print(f"Cleaned up session: {self.session_dir}")
        except:
            pass
