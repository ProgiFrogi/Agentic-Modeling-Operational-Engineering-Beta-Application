# main.py (обновлённый)
"""Main entry point for Kaggle Multi-Agent System"""

import sys
import argparse
from dotenv import load_dotenv

from config import get_config
from agents.supervisor import run_supervisor
from utils.logger import init_logger, info, error
from utils.data_downloader import KaggleDataDownloader


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description="Kaggle Multi-Agent System")
    parser.add_argument(
        "--competition",
        type=str,
        help="Competition name (overrides config)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Force download data from Kaggle"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum number of improvement iterations"
    )
    parser.add_argument(
        "--no-safe-mode",
        action="store_true",
        help="Disable safe mode (allows arbitrary code execution)"
    )
    return parser.parse_args()


def main():
    """Главная функция"""
    args = parse_args()

    # Загрузка конфигурации
    config = get_config(args.config)

    # Переопределение параметров из аргументов
    if args.competition:
        config.competition.name = args.competition
    if args.download_data:
        config.competition.download_data = True
    if args.max_iterations:
        config.pipeline.max_iterations = args.max_iterations
    if args.no_safe_mode:
        config.pipeline.safe_mode = False
        config.guardrails.enable_code_safety = False

    # Инициализация логирования
    init_logger(log_dir=config.pipeline.logs_dir)

    info("=" * 60)
    info("🚀 Starting Kaggle Competition Agent System")
    info("=" * 60)
    info(f"Competition: {config.competition.name}")
    info(f"Problem type: {config.competition.problem_type}")
    info(f"Metric: {config.competition.metric}")
    info(f"Target column: {config.competition.target_column}")
    info(f"LLM Provider: {config.model.provider}")
    info(f"Safe mode: {config.pipeline.safe_mode}")
    info(f"Max iterations: {config.pipeline.max_iterations}")

    # Загрузка данных, если нужно
    if config.competition.download_data:
        info("📥 Downloading data from Kaggle...")
        downloader = KaggleDataDownloader()
        success = downloader.download_competition_data(
            config.competition.name,
            config.competition.download_path
        )
        if not success:
            error("Failed to download data. Exiting.")
            sys.exit(1)
        info("✅ Data downloaded successfully")
    else:
        info("📁 Using local data from ./data directory")

    # Запуск супервизора
    try:
        result = run_supervisor(
            competition_name=config.competition.name,
            max_iterations=config.pipeline.max_iterations
        )

        info("=" * 60)
        if result.get("final_submission_made"):
            info("✅ Process completed with submission!")
        else:
            info("✅ Process completed!")

        info(f"Best score: {result.get('best_score', 'N/A')}")
        info(f"Total iterations: {result.get('iterations', 0)}")

    except KeyboardInterrupt:
        info("\n⚠️ Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        error(f"❌ Process failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    load_dotenv()
    main()
