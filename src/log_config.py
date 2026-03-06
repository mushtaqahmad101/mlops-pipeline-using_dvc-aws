"""
log_config.py
-------------
Centralised logging configuration for the MLOps pipeline.
Call `get_logger(__name__)` in every module to get a pre-configured logger.
"""

import logging
import os
from datetime import datetime

# ── Log directory & file ───────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)

# ── Formatters ─────────────────────────────────────────────────────────────────
LOG_FORMAT = (
    "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a logger with both console and file handlers.

    Parameters
    ----------
    name  : typically ``__name__`` of the calling module.
    level : minimum logging level (default DEBUG).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ── Console handler ────────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File handler ───────────────────────────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
