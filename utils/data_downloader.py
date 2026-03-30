"""Data downloader for Kaggle competitions"""

from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


class KaggleDataDownloader:
    """Загрузчик данных с Kaggle"""

    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()

    def download_competition_data(self, competition_name: str, download_path: str = "./data") -> bool:
        """Скачивает все файлы соревнования"""
        try:
            download_dir = Path(download_path)
            download_dir.mkdir(parents=True, exist_ok=True)

            print(f"📥 Downloading competition '{competition_name}' data to {download_dir}")

            # Скачиваем все файлы
            self.api.competition_download_files(competition_name, path=str(download_dir), quiet=False)

            # Распаковываем zip файлы
            import zipfile
            for zip_file in download_dir.glob("*.zip"):
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(download_dir)
                zip_file.unlink()  # Удаляем zip после распаковки

            print(f"✅ Data downloaded successfully to {download_dir}")
            return True

        except Exception as e:
            print(f"❌ Failed to download data: {e}")
            return False
