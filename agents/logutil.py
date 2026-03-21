"""Loggers under the ``agentic`` namespace (configure root in cli when needed)."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"agentic.{name}")
