"""Small logging helper: file + console, UTF-8 for bilingual prompts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logger(log_path: Path, name: str = "orbit") -> logging.Logger:
    """Return a logger writing to both file (UTF-8) and stdout.

    Safe to call multiple times — handlers are reset each call so the resulting
    logger reflects the newest log_path.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for existing in list(logger.handlers):
        logger.removeHandler(existing)
        existing.close()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(stream_handler)

    return logger
