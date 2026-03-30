import os
import logging
from datetime import datetime

# Глобальные переменные
_logger = None
_log_dir = "logs"
_log_filename = "session.log"

def _ensure_logger():
    """Гарантирует, что логгер инициализирован (создана папка и настроены обработчики)"""
    global _logger
    if _logger is None:
        session_time = datetime.now()
        date_str = session_time.strftime("%Y-%m-%d")          # например, 2025-03-26
        time_str = session_time.strftime("%H-%M-%S")          # например, 14-30-45
        log_dir_path = os.path.join(_log_dir, date_str, time_str)
        os.makedirs(log_dir_path, exist_ok=True)
        log_file = os.path.join(log_dir_path, _log_filename)

        _logger = logging.getLogger("KaggleAgentLogger")
        _logger.setLevel(logging.DEBUG)

        # Убираем старые обработчики, если они были (на случай повторного вызова)
        if _logger.handlers:
            _logger.handlers.clear()

        # Файловый обработчик
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        # Консольный обработчик (уровень INFO и выше)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

        # Сохраняем информацию о сессии для внешнего использования
        _logger.session_info = {
            'date': date_str,
            'time': time_str,
            'log_dir': log_dir_path,
            'log_file': log_file
        }

def init_logger(log_dir="logs", log_filename="session.log"):
    """
    Инициализирует логгер с указанными параметрами.
    Вызывать один раз в начале приложения (опционально).
    Если не вызывать, логгер создастся при первом обращении.
    """
    global _log_dir, _log_filename, _logger
    _log_dir = log_dir
    _log_filename = log_filename
    _logger = None  # сбрасываем, чтобы создать с новыми настройками
    _ensure_logger()

def debug(msg):
    _ensure_logger()
    _logger.debug(msg)

def info(msg):
    _ensure_logger()
    _logger.info(msg)

def warning(msg):
    _ensure_logger()
    _logger.warning(msg)

def error(msg):
    _ensure_logger()
    _logger.error(msg)

def critical(msg):
    _ensure_logger()
    _logger.critical(msg)

def get_session_info():
    """Возвращает информацию о текущей сессии: дату, время, папку, файл"""
    _ensure_logger()
    return _logger.session_info