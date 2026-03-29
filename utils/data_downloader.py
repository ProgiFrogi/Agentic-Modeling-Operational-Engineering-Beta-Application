# utils/data_downloader.py
"""Data downloader for Kaggle competitions"""

import os
from pathlib import Path
from typing import Optional
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

    def download_sample_submission(self, competition_name: str, output_path: Optional[str] = None) -> Optional[str]:
        """Скачивает sample submission файл"""
        try:
            files = self.api.competition_list_files(competition_name)

            for file in files:
                if "sample" in file.name.lower() and "submission" in file.name.lower():
                    output_file = output_path or file.name
                    self.api.competition_download_file(competition_name, file.name, path='.')

                    import zipfile
                    if Path(file.name).suffix == '.zip':
                        with zipfile.ZipFile(file.name, 'r') as zf:
                            zf.extractall('.')
                        Path(file.name).unlink()

                    print(f"✅ Downloaded sample submission to {output_file}")
                    return output_file

            print(f"❌ Sample submission not found for {competition_name}")
            return None

        except Exception as e:
            print(f"❌ Failed to download sample submission: {e}")
            return None